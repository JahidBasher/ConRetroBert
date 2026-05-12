"""Character-level tokenizer for SMILES and SMARTS strings.

Builds a vocabulary from individual characters observed in training data
and encodes sequences with a type-prefix token (PRODUCT or TEMPLATE).
Special tokens occupy fixed positions at the head of the vocabulary.
"""

import json
from typing import Dict, Iterable, List, Optional, Set

from .datatypes import FeatureDict


class CharTokenizer:
    """Character-level tokenizer with special tokens for SMILES/SMARTS encoding.

    Each sequence is prefixed with a type token (PRODUCT or TEMPLATE) that
    identifies the chemical role of the encoded string.  Optional BOS/EOS
    tokens can be inserted around the character body, and sequences are
    truncated to a configurable maximum length before optional right-padding.
    """

    PAD = "<PAD>"
    UNK = "<UNK>"
    MASK = "<MASK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    TEMPLATE = "TEMPLATE"
    PRODUCT = "PRODUCT"

    def __init__(self, vocab: List[str]) -> None:
        """Initialise from a pre-built vocabulary list.

        Args:
            vocab: Ordered list of token strings.  Special tokens must be
                present at the positions established by the class methods.
        """
        self.vocab = vocab
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(vocab)}
        self.itos: Dict[int, str] = {i: tok for i, tok in enumerate(vocab)}
        self.pad_id = self.stoi[self.PAD]
        self.unk_id = self.stoi[self.UNK]
        self.mask_id = self.stoi[self.MASK]
        self.template_id = self.stoi[self.TEMPLATE]
        self.product_id = self.stoi[self.PRODUCT]
        self.bos_id = self.stoi[self.BOS]
        self.eos_id = self.stoi[self.EOS]
        self.special_ids: Set[int] = {
            self.pad_id,
            self.unk_id,
            self.mask_id,
            self.template_id,
            self.product_id,
            self.bos_id,
            self.eos_id,
        }

    @staticmethod
    def preprocess(text: Optional[str]) -> str:
        """Strip all whitespace and normalise the input string.

        Args:
            text: Raw input string or None.

        Returns:
            Whitespace-free string; empty string for None input.
        """
        if text is None:
            return ""
        return "".join(str(text).split())

    @classmethod
    def build_from_texts(cls, texts: Iterable[str]) -> "CharTokenizer":
        """Build a tokenizer by scanning all unique characters in *texts*.

        Special tokens are prepended to the vocabulary; remaining characters
        are sorted lexicographically.

        Args:
            texts: Iterable of raw SMILES/SMARTS strings.

        Returns:
            A freshly constructed CharTokenizer.
        """
        chars: Set[str] = set()
        for t in texts:
            t = cls.preprocess(t)
            for ch in t:
                chars.add(ch)
        specials = [cls.PAD, cls.UNK, cls.MASK, cls.TEMPLATE, cls.PRODUCT, cls.BOS, cls.EOS]
        vocab = specials + sorted(chars)
        return cls(vocab)

    @classmethod
    def build_from_jsonl_files(cls, paths: Iterable[str], fields: Iterable[str]) -> "CharTokenizer":
        """Build a tokenizer from one or more JSONL files.

        Each line in each file is parsed as JSON and the specified fields
        are collected as training texts for the vocabulary.

        Args:
            paths: File paths to JSONL files.
            fields: JSON keys whose string values contribute to the vocabulary.

        Returns:
            A freshly constructed CharTokenizer.
        """
        texts: List[str] = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    for field in fields:
                        texts.append(obj.get(field, ""))
        return cls.build_from_texts(texts)

    def save(self, path: str) -> None:
        """Serialise the vocabulary to a JSON file.

        Args:
            path: Destination file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Deserialise a tokenizer from a JSON vocabulary file.

        Args:
            path: Path to a file previously written by :meth:`save`.

        Returns:
            Reconstructed CharTokenizer instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(obj["vocab"])

    def encode(
        self,
        text: str,
        kind: str,
        max_length: int,
        add_bos_eos: bool = False,
        pad_to_max_length: bool = False,
    ) -> FeatureDict:
        """Encode a SMILES or SMARTS string to token IDs and an attention mask.

        The sequence is prefixed with the type token for *kind*, optionally
        wrapped with BOS/EOS tokens, truncated to *max_length*, and
        optionally right-padded with PAD tokens.

        Args:
            text: Input SMILES (kind="product") or SMARTS (kind="template") string.
            kind: Either "product" or "template".
            max_length: Maximum token sequence length (inclusive of prefix/BOS/EOS).
            add_bos_eos: If True, insert BOS before and EOS after the character tokens.
            pad_to_max_length: If True, right-pad with PAD tokens to exactly *max_length*.

        Returns:
            Dict with keys "input_ids" (List[int]) and "attention_mask" (List[int]).

        Raises:
            ValueError: If *kind* is not "product" or "template".
        """
        text = self.preprocess(text)
        if kind not in ("product", "template"):
            raise ValueError(f"Unknown kind: {kind!r}")
        cls_id = self.product_id if kind == "product" else self.template_id
        tokens = [cls_id]
        if add_bos_eos:
            tokens.append(self.bos_id)
        for ch in text:
            tokens.append(self.stoi.get(ch, self.unk_id))
        if add_bos_eos:
            tokens.append(self.eos_id)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        attn = [1] * len(tokens)
        if pad_to_max_length and len(tokens) < max_length:
            pad_len = max_length - len(tokens)
            tokens = tokens + [self.pad_id] * pad_len
            attn = attn + [0] * pad_len
        return {"input_ids": tokens, "attention_mask": attn}

    def decode(self, ids: Iterable[int]) -> str:
        """Decode a sequence of token IDs back to a string, skipping special tokens.

        Args:
            ids: Iterable of integer token IDs.

        Returns:
            Reconstructed character string with all special tokens removed.
        """
        chars = []
        for i in ids:
            tok = self.itos.get(int(i), self.UNK)
            if tok in (self.PAD, self.UNK, self.MASK, self.BOS, self.EOS, self.TEMPLATE, self.PRODUCT):
                continue
            chars.append(tok)
        return "".join(chars)

    def mask_tokens(self, input_ids: List[int], mlm_prob: float) -> FeatureDict:
        """Apply random token masking for masked language modelling pre-training.

        Replaces eligible (non-special) tokens with MASK with probability
        *mlm_prob*, storing the original IDs as labels (-100 elsewhere).

        Args:
            input_ids: Sequence of token IDs to mask.
            mlm_prob: Per-token masking probability for non-special tokens.

        Returns:
            Dict with keys "input_ids" (masked sequence) and "labels"
            (original IDs at masked positions, -100 at all other positions).
        """
        import random

        labels = [-100] * len(input_ids)
        output = list(input_ids)
        for i, tok_id in enumerate(input_ids):
            if tok_id in self.special_ids:
                continue
            if random.random() < mlm_prob:
                labels[i] = tok_id
                output[i] = self.mask_id
        return {"input_ids": output, "labels": labels}
