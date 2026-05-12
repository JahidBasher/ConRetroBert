"""Stage-2 Lightning module for contrastive reaction template retrieval.

Orchestrates template embedding management, hard-negative candidate construction,
three-term listwise loss computation, and top-k retrieval metric reporting.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import BaseConRetroLightningModule
from .datatypes import CandidateSamplingConfig, FreezeConfig, Stage2Config
from .stage2_assets import Stage2TemplateAssetMixin
from .stage2_candidate_sampler import Stage2CandidateSampler
from .stage2_metrics import Stage2MetricsMixin
from .stage2_retrieval import Stage2RetrievalMixin


class EncoderFreezeScheduler:
    """Manage encoder freeze/unfreeze scheduling for Stage-2 training.

    Supports two modes:

    - **fixed**: The template and product encoders are frozen/unfrozen once at
      construction and remain constant throughout training.
    - **alternate**: The template encoder alternates between frozen and unfrozen
      states on a configurable epoch cadence; the product encoder mirrors it.

    Applying freeze state is idempotent; :meth:`apply_for_epoch` only touches
    ``requires_grad`` when the desired state differs from the current state.

    Args:
        model: Dual-encoder model whose ``template_encoder``, ``product_encoder``,
            and optional ``mlm_head`` parameters are toggled.
        freeze_cfg: Parsed freeze schedule configuration.
    """

    def __init__(self, model: Any, freeze_cfg: FreezeConfig) -> None:
        self._model = model
        self._cfg = freeze_cfg
        self._template_frozen: Optional[bool] = None
        self._product_frozen: Optional[bool] = None

    def apply_for_epoch(self, epoch: int) -> Tuple[bool, bool, bool]:
        """Apply the freeze state for *epoch* and report what changed.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            ``(template_frozen, product_frozen, changed)`` — the resolved freeze
            flags for this epoch and whether either flag changed from last call.
        """
        template_frozen, product_frozen = self._compute_freeze_state(epoch)
        changed = (
            self._template_frozen != template_frozen
            or self._product_frozen != product_frozen
        )
        if changed:
            self._apply(template_frozen, product_frozen)
            self._template_frozen = template_frozen
            self._product_frozen = product_frozen
        return template_frozen, product_frozen, changed

    def _compute_freeze_state(self, epoch: int) -> Tuple[bool, bool]:
        if not self._cfg.alternate:
            return self._cfg.template_frozen, self._cfg.product_frozen

        period = self._cfg.template_frozen_epochs + self._cfg.template_unfrozen_epochs
        if period <= 0:
            return False, False

        idx = max(0, epoch - self._cfg.start_epoch)
        window = idx % period
        if window < self._cfg.template_frozen_epochs:
            template_frozen = self._cfg.start_with_template_frozen
        else:
            template_frozen = not self._cfg.start_with_template_frozen
        return template_frozen, not template_frozen

    def _apply(self, template_frozen: bool, product_frozen: bool) -> None:
        if self._model.shared_encoder and template_frozen != product_frozen:
            raise RuntimeError(
                "Stage 2 requires model.shared_encoder=False when freezing the template "
                "encoder while leaving the product encoder trainable."
            )
        for p in self._model.template_encoder.parameters():
            p.requires_grad = not template_frozen
        for p in self._model.product_encoder.parameters():
            p.requires_grad = not product_frozen
        if self._model.mlm_head is not None:
            for p in self._model.mlm_head.parameters():
                p.requires_grad = False


class Stage2LightningModule(BaseConRetroLightningModule):
    """PyTorch Lightning module for Stage-2 contrastive training.

    Orchestrates the full Stage-2 data flow:

    1. **Asset initialisation** (``on_fit_start``): loads the template library from
       the datamodule, builds or loads normalized template embeddings, and sets up
       the configured retrieval backend.
    2. **Training step**: builds positive/hard-negative candidate sets (GPU-resident
       when a full embedding matrix is available, CPU otherwise), computes the
       three-term listwise objective ``L_rank + λ1·L_applicable + λ2·L_penalty``,
       and logs per-term losses.
    3. **Validation step**: scores candidates and computes top-k retrieval metrics
       via the ``Stage2MetricsMixin`` service.

    Args:
        cfg: Full experiment config dict.
        tokenizer: CharTokenizer or compatible duck-typed encoder.
    """

    def __init__(self, cfg: Dict[str, Any], tokenizer: Any) -> None:
        super().__init__(cfg, tokenizer)

        self._cfg = Stage2Config.from_cfg(cfg, fallback_temperature=self.temperature)
        self.candidate_sampler = Stage2CandidateSampler(seed=cfg.get("seed", 42))
        self.freeze_scheduler = EncoderFreezeScheduler(self.model, self._cfg.freeze)

        # Template assets — populated in on_fit_start when trainer/device are available.
        self.asset_manager: Optional[Stage2TemplateAssetMixin] = None
        self.retrieval: Optional[Stage2RetrievalMixin] = None
        self.metrics = Stage2MetricsMixin(self._cfg.evaluation)
        self.template_to_idx: Optional[Dict[str, int]] = None
        self.template_emb_cpu: Optional[torch.Tensor] = None
        self.template_emb_gpu: Optional[torch.Tensor] = None
        self._train_negative_template_ids: Optional[List[int]] = None
        self._train_negative_mask_cpu: Optional[torch.Tensor] = None
        self._train_negative_mask_cache: Dict[str, torch.Tensor] = {}

        # Eagerly apply initial encoder freeze so parameters are correct before
        # the first optimizer step.
        self.freeze_scheduler.apply_for_epoch(0)

    def on_fit_start(self) -> None:
        """Load Stage-2 assets before the first optimization step.

        Retrieves the template library from the datamodule, builds (or loads) the
        normalized template embedding matrix, and initializes the retrieval backend.
        Called once by PyTorch Lightning at the start of :meth:`fit`.
        """
        datamodule = self.trainer.datamodule
        template_library = getattr(datamodule, "template_library", None)
        if template_library is None:
            raise RuntimeError("Stage 2 requires datamodule.template_library.")

        self.template_list = template_library.templates
        self.template_to_idx = template_library.template_to_id
        if not self.template_list:
            raise RuntimeError("Template library is empty; cannot run Stage 2.")
        self._initialize_train_negative_template_ids(datamodule)

        self.asset_manager = Stage2TemplateAssetMixin(
            model=self.model,
            template_input_builder=self.template_input_builder,
            template_input_collator=self.template_input_collator,
            emb_cfg=self._cfg.template_embeddings,
            token_cache_cfg=self._cfg.template_token_cache,
        )
        self.template_emb_cpu = self.asset_manager.load_or_build_embeddings(
            self.template_list, self.device
        )
        if self._cfg.keep_embeddings_on_gpu:
            self.template_emb_gpu = self.template_emb_cpu.to(self.device)

        self.retrieval = Stage2RetrievalMixin(self._cfg.candidate_sampling.retrieval)
        self.retrieval.initialize(self.template_emb_cpu)

    def on_train_epoch_start(self) -> None:
        """Refresh freeze state and rebuild assets when the alternating schedule changes.

        When the freeze schedule transitions (or while the template encoder is
        unfrozen), template embeddings are rebuilt so the retrieval index stays
        aligned with the current model checkpoint.
        """
        template_frozen, _, changed = self.freeze_scheduler.apply_for_epoch(self.current_epoch)
        if self._cfg.freeze.alternate and (changed or not template_frozen):
            self.template_emb_cpu = self.asset_manager.load_or_build_embeddings(
                self.template_list, self.device, force_rebuild=True
            )
            self.template_emb_gpu = (
                self.template_emb_cpu.to(self.device)
                if self._cfg.keep_embeddings_on_gpu
                else None
            )
            self.retrieval.initialize(self.template_emb_cpu)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Execute one Stage-2 training step with the three-term objective.

        ``L = L_rank + λ1·L_applicable + λ2·L_penalty``

        - **L_rank**: multi-positive listwise NLL over observed positives with
          hard/random negatives sampled from embedding space.
        - **L_applicable**: listwise NLL over precomputed applicable templates with
          non-applicable hard/random negatives — teaches the model to rank all
          chemically valid templates above non-applicable ones.
        - **L_penalty**: mean score of top-k retrieved templates confirmed
          non-applicable by RDKit — directly suppresses the model's most confident
          chemical mistakes via absolute score minimisation.

        When GPU template embeddings are available the full ``[B, N]`` score matrix
        is computed once and reused across all three terms.  When
        ``appl_template_ids`` is absent or empty, only ``L_rank`` is computed.

        Args:
            batch: Collated training batch dict.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            Scalar total loss tensor used for backpropagation.
        """
        product_inputs = self._extract_product_inputs(batch)
        pos_template_ids: List[List[int]] = batch["pos_template_ids"]
        appl_template_ids: List[List[int]] = batch.get(
            "appl_template_ids", [[] for _ in pos_template_ids]
        )
        batch_has_applicable = any(len(a) > 0 for a in appl_template_ids)
        restrict_to_train_templates = (
            self._cfg.candidate_sampling.restrict_negatives_to_train_templates
        )
        allowed_negative_mask = (
            self._get_train_negative_mask(self.device)
            if restrict_to_train_templates
            else None
        )

        loss_cfg = self._cfg.loss
        applicable_loss_weight = loss_cfg.applicable_loss_weight
        penalty_loss_weight = loss_cfg.penalty_loss_weight
        temp = loss_cfg.temperature

        _, product_emb = self.model.encode_product(product_inputs)
        norm_product_emb = F.normalize(product_emb, dim=-1)

        if self.template_emb_gpu is not None:
            # ── Single matmul reused by all three loss terms ──────────────────
            template_emb = self.template_emb_gpu.to(dtype=norm_product_emb.dtype)
            scores_full = norm_product_emb @ template_emb.t()  # [B, N]

            # L_rank
            cand_ids, pos_mask = self.candidate_sampler.build_train_candidates_gpu(
                norm_product_emb=norm_product_emb,
                pos_template_ids=pos_template_ids,
                template_emb_gpu=self.template_emb_gpu,
                sampling_cfg=self._cfg.candidate_sampling,
                device=self.device,
                scores_full=scores_full,
                allowed_negative_mask_full=allowed_negative_mask,
            )
            scores_rank = scores_full.gather(1, cand_ids)
            if temp > 0:
                scores_rank = scores_rank / temp
            total_loss, listwise_loss, entropy = self._compute_listwise_loss(scores_rank, pos_mask)

            # L_applicable
            applicable_loss = scores_full.new_tensor(0.0)
            if batch_has_applicable and applicable_loss_weight > 0:
                appl_cand_ids, appl_pos_mask = self.candidate_sampler.build_applicable_candidates_gpu(
                    scores_full=scores_full,
                    appl_template_ids=appl_template_ids,
                    pos_template_ids=pos_template_ids,
                    sampling_cfg=self._cfg.candidate_sampling,
                    device=self.device,
                    allowed_negative_mask_full=allowed_negative_mask,
                )
                scores_appl = scores_full.gather(1, appl_cand_ids)
                if temp > 0:
                    scores_appl = scores_appl / temp
                applicable_loss, _, _ = self._compute_listwise_loss(scores_appl, appl_pos_mask)

            # L_penalty
            penalty_loss = scores_full.new_tensor(0.0)
            if batch_has_applicable and penalty_loss_weight > 0:
                penalty_loss = self._compute_penalty_loss(
                    scores_full, appl_template_ids, pos_template_ids,
                    k=loss_cfg.penalty_top_k,
                    allowed_negative_mask_full=allowed_negative_mask,
                )

            total_loss = (
                total_loss
                + applicable_loss_weight * applicable_loss
                + penalty_loss_weight * penalty_loss
            )
        else:
            # ── CPU fallback: L_rank only ─────────────────────────────────────
            cand_ids, pos_mask = self._build_training_candidates_cpu(
                norm_product_emb, pos_template_ids
            )
            scores_rank = self._score_candidate_templates(norm_product_emb, cand_ids)
            total_loss, listwise_loss, entropy = self._compute_listwise_loss(scores_rank, pos_mask)
            applicable_loss = scores_rank.new_tensor(0.0)
            penalty_loss = scores_rank.new_tensor(0.0)

        self.log("train_listwise_loss", listwise_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_entropy", entropy, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_l_applicable", applicable_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_l_penalty", penalty_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one validation batch using the CPU or GPU candidate path.

        When GPU template embeddings are available a single ``[B, N]`` matmul is
        shared across candidate selection, loss scoring, and top-k metric
        computation so there are no redundant embedding lookups or CPU transfers.

        Args:
            batch: Collated validation batch dict.
            batch_idx: Index of the current batch within the validation epoch.

        Returns:
            Scalar validation loss tensor.
        """
        product_inputs = self._extract_product_inputs(batch)
        pos_template_ids: List[List[int]] = batch["pos_template_ids"]

        _, product_emb = self.model.encode_product(product_inputs)
        norm_product_emb = F.normalize(product_emb, dim=-1)

        if self.template_emb_gpu is not None:
            template_emb = self.template_emb_gpu.to(dtype=norm_product_emb.dtype)
            scores_full = norm_product_emb @ template_emb.t()  # [B, N]

            cand_ids, pos_mask, full_pos_mask = self._build_validation_candidates_gpu(
                scores_full, pos_template_ids
            )
            scores = scores_full.gather(1, cand_ids)
            temp = self._cfg.loss.temperature
            if temp > 0:
                scores = scores / temp

            loss, listwise, entropy = self._compute_listwise_loss(scores, pos_mask)
            metrics = self.metrics.compute_gpu(scores_full, full_pos_mask)
        else:
            cand_ids, pos_mask = self._build_validation_candidates_cpu(
                norm_product_emb, pos_template_ids
            )
            scores = self._score_candidate_templates(norm_product_emb, cand_ids)
            loss, listwise, entropy = self._compute_listwise_loss(scores, pos_mask)
            metrics = self.metrics.compute_cpu(
                norm_product_emb,
                pos_template_ids,
                self.template_emb_cpu.size(0),
                self.retrieval.retrieve,
            )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_listwise_loss", listwise, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_entropy", entropy, on_step=False, on_epoch=True, prog_bar=False)
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    # ------------------------------------------------------------------
    # Candidate construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_effective_train_sampling_cfg(
        sampling_cfg: CandidateSamplingConfig,
    ) -> CandidateSamplingConfig:
        """Return a clamped candidate-sampling config for train-time candidate building.

        The effective config mirrors the input retrieval settings, but clamps
        candidate and negative budgets to old Stage-2 behavior so hard/in-batch/random
        negatives always fit inside candidate_size.
        """
        candidate_size = max(1, int(sampling_cfg.candidate_size))
        hard_negatives, inbatch_negatives, random_negatives = (
            Stage2CandidateSampler.fit_negative_budgets_to_candidate_size(
                candidate_size=candidate_size,
                hard_negatives=int(sampling_cfg.hard_negatives),
                inbatch_negatives=int(sampling_cfg.inbatch_negatives),
                random_negatives=int(sampling_cfg.random_negatives),
            )
        )
        effective_raw: Dict[str, Any] = {
            "candidate_size": candidate_size,
            "hard_negatives": hard_negatives,
            "inbatch_negatives": inbatch_negatives,
            "random_negatives": random_negatives,
            "restrict_negatives_to_train_templates": bool(
                sampling_cfg.restrict_negatives_to_train_templates
            ),
            "retrieval": dict(sampling_cfg.retrieval) if sampling_cfg.retrieval else {},
        }
        return CandidateSamplingConfig.from_dict(effective_raw)

    def _build_training_candidates_cpu(
        self,
        norm_product_emb: torch.Tensor,
        pos_template_ids: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build CPU training candidates from retrieval + in-batch + random pools.

        Args:
            norm_product_emb: L2-normalized product embeddings ``[B, D]``.
            pos_template_ids: Positive template ID lists per sample.

        Returns:
            ``(candidate_ids [B, C], positive_mask [B, C])``.
        """
        sampling_cfg = self._resolve_effective_train_sampling_cfg(
            self._cfg.candidate_sampling
        )
        retrieval_cfg = sampling_cfg.retrieval
        n_templates = self.template_emb_cpu.size(0)

        retrieval_rows: List[List[int]] = [[] for _ in range(len(pos_template_ids))]
        if retrieval_cfg.get("enabled", True) and sampling_cfg.hard_negatives > 0:
            retrieval_top_k = int(
                retrieval_cfg.get(
                    "top_k",
                    max(sampling_cfg.hard_negatives * 4, sampling_cfg.hard_negatives),
                )
            )
            allow_matrix = bool(retrieval_cfg.get("allow_matrix_in_training", False))
            retrieval_rows = self.retrieval.retrieve(
                norm_product_emb, retrieval_top_k, allow_matrix
            )

        return self.candidate_sampler.build_train_candidates_cpu(
            pos_template_ids=pos_template_ids,
            n_templates=n_templates,
            device=self.device,
            retrieval_rows=retrieval_rows,
            candidate_size=sampling_cfg.candidate_size,
            hard_negatives=sampling_cfg.hard_negatives,
            inbatch_negatives=sampling_cfg.inbatch_negatives,
            random_negatives=sampling_cfg.random_negatives,
            allowed_negative_template_ids=(
                self._train_negative_template_ids
                if sampling_cfg.restrict_negatives_to_train_templates
                else None
            ),
        )

    def _build_validation_candidates_cpu(
        self,
        norm_product_emb: torch.Tensor,
        pos_template_ids: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build CPU validation candidates from retrieval rows with random fill.

        Args:
            norm_product_emb: L2-normalized product embeddings ``[B, D]``.
            pos_template_ids: Positive template ID lists per sample.

        Returns:
            ``(candidate_ids [B, C], positive_mask [B, C])``.
        """
        eval_cfg = self._cfg.evaluation
        n_templates = self.template_emb_cpu.size(0)
        retrieval_rows = self.retrieval.retrieve(
            norm_product_emb, eval_cfg.retrieval_top_k, allow_matrix=True
        )
        return self.candidate_sampler.build_eval_candidates_cpu(
            pos_template_ids=pos_template_ids,
            n_templates=n_templates,
            device=self.device,
            retrieval_rows=retrieval_rows,
            candidate_size=eval_cfg.candidate_size,
        )

    def _build_validation_candidates_gpu(
        self,
        scores_full: torch.Tensor,
        pos_template_ids: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build GPU validation candidate IDs and masks from the full score matrix.

        Args:
            scores_full: Full ``[B, N]`` similarity score matrix.
            pos_template_ids: Positive template ID lists per sample.

        Returns:
            ``(candidate_ids [B, C], positive_mask [B, C], full_positive_mask [B, N])``.
        """
        return self.candidate_sampler.build_eval_candidates_gpu(
            scores_full=scores_full,
            pos_template_ids=pos_template_ids,
            candidate_size=self._cfg.evaluation.candidate_size,
        )

    def _score_candidate_templates(
        self,
        norm_product_emb: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temperature-scaled scores for selected candidate templates.

        Resolves candidate embeddings with a unique-index cache on the CPU path
        to avoid redundant index operations, then applies temperature scaling.

        Args:
            norm_product_emb: L2-normalized product embeddings ``[B, D]``.
            candidate_ids: Selected template IDs ``[B, C]``.

        Returns:
            Score tensor ``[B, C]``.
        """
        batch_size, candidate_count = candidate_ids.shape
        if self.template_emb_gpu is not None:
            cand_emb = self.template_emb_gpu[candidate_ids]
        else:
            flat_idx = candidate_ids.view(-1).detach().cpu()
            uniq_idx, inverse = torch.unique(flat_idx, sorted=False, return_inverse=True)
            uniq_emb = self.template_emb_cpu.index_select(0, uniq_idx).to(self.device)
            cand_emb = uniq_emb[inverse].view(batch_size, candidate_count, -1)

        cand_emb = cand_emb.to(dtype=norm_product_emb.dtype)
        scores = (norm_product_emb.unsqueeze(1) * cand_emb).sum(dim=-1)

        temp = self._cfg.loss.temperature
        if temp > 0:
            scores = scores / temp
        return scores

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_listwise_loss(
        self,
        scores: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute multi-positive listwise NLL with optional label smoothing and entropy bonus.

        Args:
            scores: Candidate scores ``[B, C]`` (temperature already applied).
            positive_mask: Boolean mask ``[B, C]`` indicating positive candidates.

        Returns:
            ``(total_loss, listwise_loss, entropy)`` — total loss optionally
            subtracts ``entropy_bonus * entropy`` from the base listwise term.
        """
        log_probs = F.log_softmax(scores, dim=-1)
        pos_counts = positive_mask.sum(dim=-1).clamp_min(1)

        label_smoothing = self._cfg.loss.label_smoothing
        entropy_bonus = self._cfg.loss.entropy_bonus

        if label_smoothing > 0.0:
            candidate_size = scores.size(1)
            target = torch.full_like(
                log_probs, fill_value=label_smoothing / float(candidate_size)
            )
            target = target + positive_mask.float() * (
                (1.0 - label_smoothing) / pos_counts.float()
            ).unsqueeze(-1)
            nll = -(target * log_probs).sum(dim=-1)
        else:
            pos_log_probs = torch.where(
                positive_mask, log_probs, torch.zeros_like(log_probs)
            )
            nll = -(pos_log_probs.sum(dim=-1) / pos_counts.float())

        listwise = nll.mean()
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        total = listwise - entropy_bonus * entropy
        return total, listwise, entropy

    def _compute_penalty_loss(
        self,
        scores_full: torch.Tensor,
        appl_template_ids: List[List[int]],
        pos_template_ids: List[List[int]],
        k: int = 512,
        allowed_negative_mask_full: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute L_penalty: mean score of top-k retrieved templates that are non-applicable.

        ``F(x) = top-k(x) ∩ (T \\ T(x))``

        For each sample that has applicable templates, find the top-k retrieved
        templates that RDKit confirms cannot react with the product.  Minimising
        their mean score provides an absolute (non-relative) suppression signal
        that complements the relative softmax signal from L_applicable.

        Args:
            scores_full: Full ``[B, N]`` similarity score matrix.
            appl_template_ids: Precomputed applicable template IDs per sample.
            pos_template_ids: Observed positive template IDs per sample.
            k: Number of top-scoring templates to inspect per sample.

        Returns:
            Scalar penalty loss tensor (zero when no non-applicable top-k exists).
        """
        B, N = scores_full.shape
        device = scores_full.device

        # T(x) = appl ∪ pos — anything applicable must not be penalized.
        all_applicable = [
            sorted(set(a) | set(p)) for a, p in zip(appl_template_ids, pos_template_ids)
        ]
        appl_mask = Stage2CandidateSampler.build_positive_template_mask(
            all_applicable, N, device
        )  # [B, N]

        scores_for_topk = scores_full
        if allowed_negative_mask_full is not None:
            allowed_mask = allowed_negative_mask_full.to(
                device=device, dtype=torch.bool, non_blocking=True
            )
            scores_for_topk = scores_full.masked_fill(
                (~allowed_mask).unsqueeze(0), float("-inf")
            )

        topk = min(k, N)
        _, topk_ids = torch.topk(scores_for_topk, k=topk, dim=-1)  # [B, topk]
        is_nonappl = ~appl_mask.gather(1, topk_ids)  # [B, topk]

        retrieved_scores = scores_for_topk.gather(1, topk_ids)  # [B, topk]
        finite_mask = torch.isfinite(retrieved_scores)
        is_nonappl = is_nonappl & finite_mask
        nonappl_scores = retrieved_scores * is_nonappl.float()
        nonappl_count = is_nonappl.float().sum(dim=-1).clamp_min(1.0)
        penalty_per_sample = nonappl_scores.sum(dim=-1) / nonappl_count  # [B]

        has_appl = torch.tensor(
            [len(a) > 0 for a in appl_template_ids], dtype=torch.bool, device=device
        )
        has_nonappl = is_nonappl.any(dim=-1) & has_appl

        if not has_nonappl.any():
            return scores_full.new_tensor(0.0)
        return penalty_per_sample[has_nonappl].mean()

    def _initialize_train_negative_template_ids(self, datamodule: Any) -> None:
        """Build train-template ID allowlist for strict train-only negative mining."""
        self._train_negative_template_ids = None
        self._train_negative_mask_cpu = None
        self._train_negative_mask_cache = {}
        if not self._cfg.candidate_sampling.restrict_negatives_to_train_templates:
            return

        train_dataset = getattr(datamodule, "train_dataset", None)
        samples = getattr(train_dataset, "samples", None)
        if not isinstance(samples, list):
            raise RuntimeError(
                "Stage 2 train-only negative mining requires datamodule.train_dataset.samples."
            )

        n_templates = len(self.template_list)
        train_ids = set()
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            for raw_tid in sample.get("pos_template_ids", []):
                try:
                    tid = int(raw_tid)
                except Exception:
                    continue
                if 0 <= tid < n_templates:
                    train_ids.add(tid)

        if not train_ids:
            raise RuntimeError(
                "Stage 2 train-only negative mining found no template IDs in train split."
            )

        self._train_negative_template_ids = sorted(train_ids)
        train_negative_mask = torch.zeros(n_templates, dtype=torch.bool)
        train_negative_mask[self._train_negative_template_ids] = True
        self._train_negative_mask_cpu = train_negative_mask
        self._train_negative_mask_cache = {}
        if getattr(self.trainer, "is_global_zero", True):
            print(
                "[stage2.train_only_negatives] enabled with "
                f"{len(self._train_negative_template_ids)} / {n_templates} templates "
                "available for train-time negatives."
            )

    def _get_train_negative_mask(self, device: torch.device) -> Optional[torch.Tensor]:
        """Return cached train-template allow-mask on the target device."""
        if self._train_negative_mask_cpu is None:
            return None
        device_key = f"{device.type}:{device.index}"
        cached = self._train_negative_mask_cache.get(device_key)
        if cached is None:
            cached = self._train_negative_mask_cpu.to(
                device=device,
                dtype=torch.bool,
                non_blocking=True,
            )
            self._train_negative_mask_cache[device_key] = cached
        return cached
