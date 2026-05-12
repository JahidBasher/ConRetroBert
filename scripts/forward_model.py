"""
Forward-synthesis model wrapper (Molecular Transformer via OpenNMT-py==0.4.1).

Usage
-----
    model = ForwardModel("path/to/mol_former.pt")

    # predict products from a list of reactant SMILES
    products = model.predict(["Nc1ccccc1.O=C(Cl)c1ccccc1", "CC(=O)O.Oc1ccccc1"])

    # predict and attach to a DataFrame (adds 'pred_product' column)
    df = model.predict_df(df)
"""

import argparse
import logging
import re

from rdkit import Chem
import onmt.inputters as inputters
import onmt.model_builder
import onmt.opts as opts
import torch
from onmt.translate.translator import Translator
from onmt.utils.misc import tile

logger = logging.getLogger(__name__)


class ForwardModel:
    """
    Molecular Transformer forward-synthesis model.

    Loads a pre-trained mol_former.pt checkpoint once and exposes two
    prediction methods. Load once, call many times.

    Parameters
    ----------
    model_path : str or Path
        Path to the mol_former.pt checkpoint.
    n_best : int
        Number of beam-search hypotheses to generate. Top-1 is returned.
    batch_size : int
        Inference batch size.

    Examples
    --------
    >>> model = ForwardModel("mol_former.pt")
    >>> model.predict(["Nc1ccccc1.O=C(Cl)c1ccccc1"])
    ['O=C(Nc1ccccc1)c1ccccc1']
    >>> df = model.predict_df(df)   # adds 'pred_product' column
    """

    def __init__(self, model_path, n_best=5, batch_size=128, device=-1):
        self.model_path = str(model_path)
        self.n_best = n_best
        self.batch_size = batch_size
        self.device = device
        logger.info("Loading forward model from: %s", model_path)
        self._translator = self._load(model_path, n_best, device)
        logger.info("Forward model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, smiles_list):
        """
        Predict the product for each reactant SMILES string.

        Parameters
        ----------
        smiles_list : list of str
            Reactant SMILES strings (atom-mapped or plain).

        Returns
        -------
        list of str
            Top-1 predicted product SMILES, one per input.
        """
        tokenized = [self.tokenize(self.strip_atom_map(s.strip())) for s in smiles_list]
        _, preds = self._translator.translate(
            src_data_iter=tokenized,
            batch_size=self.batch_size,
            attn_debug=False,
        )
        return ["".join(p[0].strip().split()) for p in preds]

    def predict_df(self, df):
        """
        Run predictions on a DataFrame and attach results as 'pred_product'.

        Deduplicates SMILES before inference for efficiency.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'pred' column of reactant SMILES strings.

        Returns
        -------
        pd.DataFrame
            Copy of df with a new 'pred_product' column.
        """
        unique = list(set(df["pred"]))
        pred_map = dict(zip(unique, self.predict(unique)))
        result = df.copy()
        result["pred_product"] = result["pred"].map(pred_map)
        return result

    @staticmethod
    def strip_atom_map(smi):
        """Remove atom-map numbers so the Molecular Transformer sees plain SMILES."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol)

    @staticmethod
    def tokenize(smi):
        """
        Tokenize a SMILES string into space-separated tokens.

        Reference: https://github.com/pschwllr/MolecularTransformer
        """
        pattern = (
            r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
            r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        )
        tokens = re.compile(pattern).findall(smi)
        if smi != "".join(tokens):
            logger.warning("tokenize: token mismatch for: %s", smi)
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Internals — OpenNMT loader (do not edit unless upgrading OpenNMT-py)
    # ------------------------------------------------------------------

    @staticmethod
    def _load(model_path, n_best, device=-1):
        args = argparse.Namespace(
            models=[str(model_path)],
            n_best=n_best,
            src="input.txt",
            output="pred.txt",
            batch_size=128,
            replace_unk=True,
            max_length=200,
            fast=True,
            data_type="text",
            alpha=0.0,
            beta=-0.0,
            block_ngram_repeat=0,
            ignore_when_blocking=[],
            length_penalty="none",
            coverage_penalty="none",
            stepwise_penalty=False,
            beam_size=5,
            min_length=0,
            dump_beam="",
            verbose=False,
            report_bleu=False,
            gpu=device,
            sample_rate=16000,
            window_size=0.02,
            window_stride=0.01,
            window="hamming",
            image_channel_size=3,
            attn_debug=False,
        )
        out_file = open(args.output, "w+", encoding="utf-8")
        dummy_parser = argparse.ArgumentParser(description="train.py")
        opts.model_opts(dummy_parser)
        dummy_opt = dummy_parser.parse_known_args([])[0]

        if len(args.models) > 1:
            fields, model, model_opt = onmt.decoders.ensemble.load_test_model(
                args, dummy_opt.__dict__
            )
        else:
            fields, model, model_opt = onmt.model_builder.load_test_model(
                args, dummy_opt.__dict__
            )

        scorer = onmt.translate.GNMTGlobalScorer(
            args.alpha, args.beta, args.coverage_penalty, args.length_penalty
        )
        kwargs = {
            k: getattr(args, k)
            for k in [
                "beam_size",
                "n_best",
                "max_length",
                "min_length",
                "stepwise_penalty",
                "block_ngram_repeat",
                "ignore_when_blocking",
                "dump_beam",
                "report_bleu",
                "data_type",
                "replace_unk",
                "gpu",
                "verbose",
                "fast",
                "sample_rate",
                "window_size",
                "window_stride",
                "window",
                "image_channel_size",
            ]
        }
        return _ModifiedTranslator(
            model,
            fields,
            global_scorer=scorer,
            out_file=out_file,
            report_score=False,
            copy_attn=model_opt.copy_attn,
            logger=logger,
            **kwargs
        )


# ---------------------------------------------------------------------------
# OpenNMT beam-search override — same logic as original, do not modify
# ---------------------------------------------------------------------------


class _ModifiedTranslator(Translator):

    def _fast_translate_batch(
        self, batch, data, max_length, min_length=0, n_best=1, return_attention=False
    ):
        assert data.data_type == "text"
        assert not self.copy_attn
        assert not self.dump_beam
        assert not self.use_filter_pred
        assert self.block_ngram_repeat == 0
        assert self.global_scorer.beta == 0

        beam_size, batch_size = self.beam_size, batch.batch_size
        vocab = self.fields["tgt"].vocab
        start_token = vocab.stoi[inputters.BOS_WORD]
        end_token = vocab.stoi[inputters.EOS_WORD]

        src = inputters.make_features(batch, "src", data.data_type)
        _, src_lengths = batch.src
        enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
            src, memory_bank, enc_states, with_cache=True
        )

        dec_states.map_batch_fn(lambda state, dim: tile(state, beam_size, dim=dim))
        memory_bank = tile(memory_bank, beam_size, dim=1)
        memory_lengths = tile(src_lengths, beam_size)

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=memory_bank.device
        )
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=memory_bank.device,
        )
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            start_token,
            dtype=torch.long,
            device=memory_bank.device,
        )
        alive_attn = None
        topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (beam_size - 1), device=memory_bank.device
        ).repeat(batch_size)

        hypotheses = [[] for _ in range(batch_size)]
        results = {
            "predictions": [[] for _ in range(batch_size)],
            "scores": [[] for _ in range(batch_size)],
            "attention": [[] for _ in range(batch_size)],
            "gold_score": [0] * batch_size,
            "batch": batch,
        }

        for step in range(max_length):
            dec_out, dec_states, attn = self.model.decoder(
                alive_seq[:, -1].view(1, -1, 1),
                memory_bank,
                dec_states,
                memory_lengths=memory_lengths,
                step=step,
            )
            log_probs = self.model.generator.forward(dec_out.squeeze(0))
            vocab_size = log_probs.size(-1)
            if step < min_length:
                log_probs[:, end_token] = -1e20
            log_probs += topk_log_probs.view(-1).unsqueeze(1)
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** self.global_scorer.alpha
            curr_scores = (log_probs / length_penalty).reshape(
                -1, beam_size * vocab_size
            )
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)
            topk_log_probs = topk_scores * length_penalty
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)
            batch_index = (
                topk_beam_index + beam_offset[: topk_beam_index.size(0)].unsqueeze(1)
            ).to(torch.long)
            select_indices = batch_index.view(-1).to(torch.long)
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)],
                dim=-1,
            )

            if return_attention:
                current_attn = attn["std"].index_select(1, select_indices)
                alive_attn = (
                    current_attn
                    if alive_attn is None
                    else torch.cat(
                        [alive_attn.index_select(1, select_indices), current_attn],
                        dim=0,
                    )
                )

            is_finished = topk_ids.eq(end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            end_condition = is_finished[:, 0].eq(1)

            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                attention = (
                    alive_attn.view(
                        alive_attn.size(0), -1, beam_size, alive_attn.size(-1)
                    )
                    if alive_attn is not None
                    else None
                )
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    for j in is_finished[i].nonzero().view(-1):
                        hypotheses[b].append(
                            (
                                topk_scores[i, j],
                                predictions[i, j, 1:],
                                (
                                    attention[:, i, j, : memory_lengths[i]]
                                    if attention is not None
                                    else None
                                ),
                            )
                        )
                    if end_condition[i]:
                        for n, (score, pred, attn_) in enumerate(
                            sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        ):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                            results["attention"][b].append(
                                attn_ if attn_ is not None else []
                            )

                non_finished = end_condition.eq(0).nonzero().view(-1)
                if len(non_finished) == 0:
                    break
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(
                    -1, alive_seq.size(-1)
                )
                if alive_attn is not None:
                    alive_attn = attention.index_select(1, non_finished).view(
                        alive_attn.size(0), -1, alive_attn.size(-1)
                    )

            select_indices = batch_index.view(-1)
            memory_bank = memory_bank.index_select(1, select_indices)
            memory_lengths = memory_lengths.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices)
            )

        return results


if __name__ == "__main__":


    def read_jsonl(path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
        
    import pandas as pd

    model = ForwardModel("models/STEREO_separated_augm_model_average_20.pt")
    print("Predicting products for sample reactants...")
    print(model.predict(["Nc1ccccc1.O=C(Cl)c1ccccc1"]))
    data = read_jsonl("data/uspto-50k/raw_val_valid_only.jsonl")
    reactants = [r['reactants'] for r in data]

    #timeit
    import time
    start_time = time.time()

    batch_size = 32
    num_correct = 0
    import tqdm
    for i in tqdm.tqdm(range(0, len(reactants), batch_size)):
        batch_start_time = time.time()
        batch = reactants[i:i+batch_size]
        df = pd.DataFrame({"pred": batch})
        df = model.predict_df(df)
        df['GT'] = [r['product'] for r in data[i:i+batch_size]]
        df['GT_canon'] = df['GT'].apply(ForwardModel.strip_atom_map)
        df['pred_canon'] = df['pred_product'].apply(ForwardModel.strip_atom_map)
        df['True'] = df['GT_canon'] == df['pred_canon']
        num_correct += df['True'].sum()
        print(df)
        end_time = time.time()
        print(f"Time taken for batch: {end_time - batch_start_time} seconds")
          # remove this break to run on all batches

    end_time = time.time()
    accuracy = num_correct / len(reactants)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Time taken: {end_time - start_time} seconds")
