"""Stage-2 EMA Lightning module.

Extends Stage-2 with three interlocking features:

1. **Trainable template encoder** — TE is unfrozen (fully or partially) and trained
   with a separate AdamW optimizer, preventing the stale-negative drift that occurs
   when TE is permanently frozen while PE improves.

2. **EMA stabilization** — an exponential moving average shadow of TE is maintained
   and used to rebuild the template embedding cache at epoch start, keeping the
   retrieval index smooth even as live TE parameters change.

3. **Manual optimization** — ``automatic_optimization = False`` lets the training
   step selectively step only the phase-active optimizer(s) and apply gradient
   accumulation independently per phase.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import BaseConRetroLightningModule
from ..stage_two.datatypes import CandidateSamplingConfig, FeatureCache
from ..stage_two.stage2 import EncoderFreezeScheduler, Stage2LightningModule
from ..stage_two.stage2_candidate_sampler import Stage2CandidateSampler
from .datatypes import EmaFreezeConfig, Stage2EmaConfig, TrainableTemplateConfig
from .stage2_ema import Stage2EMAMixin


# ---------------------------------------------------------------------------
# Extended freeze scheduler
# ---------------------------------------------------------------------------


class EmaEncoderFreezeScheduler(EncoderFreezeScheduler):
    """Freeze scheduler with independent product-encoder control per phase.

    Extends :class:`~stage_two.stage2.EncoderFreezeScheduler` with two config
    fields that let the product encoder be frozen or unfrozen independently of
    the template encoder in each alternating phase.

    Also handles partial template-encoder unfreeze (last-k layers), per-layer
    gradient checkpointing, and projector freeze policies.

    Args:
        model: Dual-encoder model.
        freeze_cfg: Extended freeze schedule including per-phase product flags.
        trainable_cfg: Trainable-template phase settings (partial unfreeze, etc.).
    """

    def __init__(
        self,
        model: Any,
        freeze_cfg: EmaFreezeConfig,
        trainable_cfg: TrainableTemplateConfig,
    ) -> None:
        # Pass a compatible FreezeConfig-shaped object to the parent.
        # We store the full EmaFreezeConfig ourselves for the overrides.
        super().__init__(model, freeze_cfg)  # type: ignore[arg-type]
        self._ema_freeze_cfg = freeze_cfg
        self._trainable_cfg = trainable_cfg

    # Override: per-phase product freeze
    def _compute_freeze_state(self, epoch: int) -> Tuple[bool, bool]:
        cfg = self._ema_freeze_cfg
        if not cfg.alternate:
            return cfg.template_frozen, cfg.product_frozen

        period = cfg.template_frozen_epochs + cfg.template_unfrozen_epochs
        if period <= 0:
            return False, False

        idx = max(0, epoch - cfg.start_epoch)
        window = idx % period
        template_frozen = (
            cfg.start_with_template_frozen
            if window < cfg.template_frozen_epochs
            else not cfg.start_with_template_frozen
        )
        product_frozen = (
            cfg.product_frozen_while_template_frozen
            if template_frozen
            else cfg.product_frozen_while_template_unfrozen
        )
        return template_frozen, product_frozen

    # Override: richer template tower policy
    def _apply(self, template_frozen: bool, product_frozen: bool) -> None:
        if self._model.shared_encoder and template_frozen != product_frozen:
            raise RuntimeError(
                "Stage-2 EMA requires model.shared_encoder=False when freezing the "
                "template encoder while leaving the product encoder trainable."
            )
        # Product encoder
        for p in self._model.product_encoder.parameters():
            p.requires_grad = not product_frozen
        # Template encoder (with optional partial-unfreeze logic)
        self._apply_template_tower(template_frozen)
        # MLM head always frozen in Stage-2
        if self._model.mlm_head is not None:
            for p in self._model.mlm_head.parameters():
                p.requires_grad = False

    def _apply_template_tower(self, template_frozen: bool) -> None:
        tcfg = self._trainable_cfg
        if template_frozen:
            self._set_requires_grad(self._model.template_encoder, False)
            if tcfg.freeze_projector_with_template_encoder:
                self._set_requires_grad(
                    getattr(self._model, "template_projector", None), False
                )
            self._toggle_gradient_checkpointing(False)
            return

        # Trainable phase: full or last-k-layers partial unfreeze
        if tcfg.train_last_k_layers is None:
            self._set_requires_grad(self._model.template_encoder, True)
        else:
            self._apply_last_k_layer_unfreeze(tcfg.train_last_k_layers)

        if tcfg.train_template_projector is not None:
            self._set_requires_grad(
                getattr(self._model, "template_projector", None),
                tcfg.train_template_projector,
            )
        self._toggle_gradient_checkpointing(tcfg.gradient_checkpointing)

    def _apply_last_k_layer_unfreeze(self, k: int) -> None:
        self._set_requires_grad(self._model.template_encoder, False)
        layers = self._get_transformer_layers()
        if layers is None:
            # Fallback for custom encoders without standard layer list.
            self._set_requires_grad(self._model.template_encoder, True)
            return
        for layer in layers[-min(max(1, k), len(layers)) :]:
            self._set_requires_grad(layer, True)
        if self._trainable_cfg.train_template_embeddings:
            for name in ("token_emb", "pos_emb", "norm"):
                self._set_requires_grad(
                    getattr(self._model.template_encoder, name, None), True
                )

    def _get_transformer_layers(self) -> Optional[List[Any]]:
        encoder = getattr(self._model, "template_encoder", None)
        transformer = getattr(encoder, "encoder", None)
        layers = getattr(transformer, "layers", None)
        return list(layers) if layers is not None else None

    def _toggle_gradient_checkpointing(self, enabled: bool) -> None:
        encoder = getattr(self._model, "template_encoder", None)
        if encoder is not None and hasattr(encoder, "set_gradient_checkpointing"):
            encoder.set_gradient_checkpointing(enabled)

    @staticmethod
    def _set_requires_grad(module: Any, requires_grad: bool) -> None:
        if module is None or not hasattr(module, "parameters"):
            return
        for p in module.parameters():
            p.requires_grad = requires_grad


# ---------------------------------------------------------------------------
# Trainable-template scoring service
# ---------------------------------------------------------------------------


class Stage2TrainableTemplateMixin:
    """On-the-fly template encoding and scoring for the trainable-TE phase.

    During the trainable-template phase, candidate templates cannot be scored
    from the frozen embedding cache — they must be encoded live to keep
    gradients flowing through TE.  This service encapsulates that logic:
    tokenization, chunked encoding, and dot-product scoring.

    This class is a standalone service injected into
    :class:`Stage2EmaLightningModule`; the name follows the package convention.

    Args:
        model: Dual-encoder model with ``encode_template`` method.
        template_input_builder: Callable that tokenizes a single template string.
        template_input_collator: Callable that batches feature dicts.
        trainable_cfg: Trainable-template phase settings (encode batch size, etc.).
    """

    def __init__(
        self,
        model: Any,
        template_input_builder: Any,
        template_input_collator: Any,
        trainable_cfg: TrainableTemplateConfig,
    ) -> None:
        self._model = model
        self._template_input_builder = template_input_builder
        self._template_input_collator = template_input_collator
        self._cfg = trainable_cfg

    def score_candidates(
        self,
        norm_product_emb: torch.Tensor,
        candidate_ids: torch.Tensor,
        feature_cache: Optional[FeatureCache],
        template_list: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode unique candidate templates live and return dot-product scores.

        Deduplicates candidate IDs across the batch before encoding to avoid
        redundant forward passes through TE.

        Args:
            norm_product_emb: L2-normalized product embeddings ``[B, D]``.
            candidate_ids: Candidate template IDs ``[B, C]``.
            feature_cache: Optional pre-tokenized feature cache for fast input building.
            template_list: Ordered list of SMARTS template strings.
            device: Target device.

        Returns:
            Raw (un-temperature-scaled) score tensor ``[B, C]``.
        """
        batch_size, cand_count = candidate_ids.shape
        flat_ids = candidate_ids.reshape(-1).detach().cpu().long()
        uniq_ids, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
        uniq_emb = self.encode_unique_templates(uniq_ids, feature_cache, template_list, device)
        inverse = inverse.to(device=device)
        cand_emb = uniq_emb[inverse].view(batch_size, cand_count, -1)
        return self.score_from_embeddings(norm_product_emb, cand_emb)

    def encode_unique_templates(
        self,
        uniq_ids: torch.Tensor,
        feature_cache: Optional[FeatureCache],
        template_list: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a deduplicated set of templates through live TE (keeping gradients).

        Args:
            uniq_ids: 1-D tensor of unique template IDs to encode.
            feature_cache: Optional pre-tokenized feature cache.
            template_list: Ordered list of SMARTS template strings.
            device: Target device.

        Returns:
            Embedding tensor ``[len(uniq_ids), D]``.
        """
        inputs = self.build_inputs(uniq_ids, feature_cache, template_list, device)
        return self._encode_in_chunks(inputs)

    def build_inputs(
        self,
        uniq_ids: torch.Tensor,
        feature_cache: Optional[FeatureCache],
        template_list: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Build tokenized inputs for unique template IDs.

        Uses the feature cache when valid (fast path), otherwise falls back to
        the input builder + collator.

        Args:
            uniq_ids: 1-D tensor of unique integer template IDs.
            feature_cache: Optional pre-tokenized tensors aligned to template_list.
            template_list: Ordered list of SMARTS template strings.
            device: Target device.

        Returns:
            ``Dict[str, torch.Tensor]`` ready for ``model.encode_template``.
        """
        if self._is_valid_cache(feature_cache, len(template_list)):
            return {
                k: v[uniq_ids].to(
                    dtype=torch.long if k in ("input_ids", "attention_mask") else v.dtype,
                    device=device,
                )
                for k, v in feature_cache.items()
                if isinstance(v, torch.Tensor) and v.shape[0] == len(template_list)
            }
        texts = [template_list[int(x)] for x in uniq_ids.tolist()]
        features = [self._template_input_builder(t) for t in texts]
        inputs = self._template_input_collator(features)
        if not isinstance(inputs, dict):
            raise RuntimeError("Template input collator must return a dict.")
        return {k: v.to(device=device) for k, v in inputs.items()}

    @staticmethod
    def score_from_embeddings(
        norm_product_emb: torch.Tensor,
        cand_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-candidate dot-product scores.

        Args:
            norm_product_emb: ``[B, D]``
            cand_emb: ``[B, C, D]``

        Returns:
            Score tensor ``[B, C]``.
        """
        return (norm_product_emb.unsqueeze(1) * cand_emb).sum(dim=-1)

    def _encode_in_chunks(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = next(iter(inputs.values())).shape[0]
        chunks: List[torch.Tensor] = []
        for start in range(0, total, self._cfg.encode_batch_size):
            end = min(start + self._cfg.encode_batch_size, total)
            chunk = {k: v[start:end] for k, v in inputs.items()}
            _, emb = self._model.encode_template(chunk)
            chunks.append(emb)
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _is_valid_cache(cache: Optional[FeatureCache], n_templates: int) -> bool:
        if cache is None:
            return False
        return all(
            isinstance(cache.get(k), torch.Tensor)
            and cache[k].shape[0] == n_templates
            and cache[k].dim() >= 1
            for k in ("input_ids", "attention_mask")
        )


# ---------------------------------------------------------------------------
# Stage-2 EMA Lightning module
# ---------------------------------------------------------------------------


class Stage2EmaLightningModule(Stage2LightningModule):
    """Stage-2 with trainable template encoder, EMA stabilization, and manual optimization.

    Inherits all frozen-TE logic from :class:`~stage_two.stage2.Stage2LightningModule`
    and adds:

    - A second AdamW optimizer for the template tower (stepped only during
      template-trainable epochs to keep Adam moment estimates clean).
    - An EMA shadow of TE rebuilt into the embedding cache at each epoch start.
    - Live template encoding for candidate scoring when TE is trainable (keeping
      gradients through TE while the product side uses the frozen cache).
    - Optional micro-batching and last-k-layer partial unfreeze for memory safety.

    Args:
        cfg: Full experiment config dict.
        tokenizer: CharTokenizer or compatible duck-typed encoder.
    """

    # Disable Lightning's automatic optimizer/scheduler stepping.
    automatic_optimization = False

    def __init__(self, cfg: Dict[str, Any], tokenizer: Any) -> None:
        super().__init__(cfg, tokenizer)

        opt_lr = float(cfg.get("optimizer", {}).get("lr", 1e-4))
        self._ema_cfg = Stage2EmaConfig.from_cfg(cfg, base_lr=opt_lr)

        # Replace parent's freeze_scheduler with the EMA-aware version.
        self.freeze_scheduler = EmaEncoderFreezeScheduler(
            self.model, self._ema_cfg.freeze, self._ema_cfg.trainable_template
        )
        self.freeze_scheduler.apply_for_epoch(0)

        # EMA service and trainable scorer — created in on_fit_start.
        self.ema_encoder: Optional[Stage2EMAMixin] = None
        self.trainable_scorer: Optional[Stage2TrainableTemplateMixin] = None
        self.feature_cache: Optional[FeatureCache] = None

        # Dual-optimizer references — populated in configure_optimizers.
        self._opt_product: Optional[torch.optim.Optimizer] = None
        self._opt_template: Optional[torch.optim.Optimizer] = None
        self._sched_product: Optional[Any] = None
        self._sched_product_interval: str = "step"
        self._sched_template: Optional[Any] = None
        self._nan_guard_enabled: bool = bool(self._ema_cfg.skip_non_finite_batches)
        self._non_finite_loss_skip_count: int = 0
        self._non_finite_grad_skip_count: int = 0

    # ------------------------------------------------------------------
    # Optimizer setup
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Any:
        """Return two independent AdamW optimizers, one per encoder tower.

        Keeping optimizer state separate prevents the "LR spike on unfreeze"
        problem where frozen params accumulate near-zero second moments that
        corrupt the first real update after unfreezing.

        Only the product optimizer receives an LR scheduler (same warmup-cosine
        logic as Stage-1).  The template optimizer uses a fixed lower LR; its
        Adam state is optionally cleared on entry to each trainable phase.
        """
        if self.model.shared_encoder:
            result = BaseConRetroLightningModule.configure_optimizers(self)
            shared_opt = result["optimizer"] if isinstance(result, dict) else result
            self._opt_product = shared_opt
            if isinstance(result, dict) and "lr_scheduler" in result:
                lr_dict = result["lr_scheduler"]
                self._sched_product = lr_dict["scheduler"]
                self._sched_product_interval = lr_dict.get("interval", "step")
            return result

        opt_cfg = self.cfg["optimizer"]
        base_lr = float(opt_cfg["lr"])
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))
        template_lr = self._ema_cfg.template_lr

        product_params: List[torch.nn.Parameter] = list(
            self.model.product_encoder.parameters()
        )
        product_proj = getattr(self.model, "product_projector", None)
        if product_proj is not None:
            product_params += list(product_proj.parameters())
        if not product_params:
            raise RuntimeError("No product encoder parameters found for Stage-2 EMA optimizer.")

        template_params: List[torch.nn.Parameter] = list(
            self.model.template_encoder.parameters()
        )
        template_proj = getattr(self.model, "template_projector", None)
        if template_proj is not None:
            if self._ema_cfg.trainable_template.train_template_projector is not False:
                template_params += list(template_proj.parameters())
        if not template_params:
            raise RuntimeError("No template encoder parameters found for Stage-2 EMA optimizer.")

        opt_product = torch.optim.AdamW(
            product_params, lr=base_lr, weight_decay=weight_decay
        )
        opt_template = torch.optim.AdamW(
            template_params, lr=template_lr, weight_decay=weight_decay
        )
        self._opt_product = opt_product
        self._opt_template = opt_template

        te_warmup = self._ema_cfg.template_warmup_steps
        if te_warmup > 0:
            def _te_warmup_lambda(step: int) -> float:
                return min(1.0, (step + 1) / te_warmup)
            self._sched_template = torch.optim.lr_scheduler.LambdaLR(
                opt_template, lr_lambda=_te_warmup_lambda
            )

        sched_result = self._build_lr_scheduler(opt_product)
        if sched_result is not None:
            sched, interval = sched_result
            self._sched_product = sched
            self._sched_product_interval = interval
            return (
                [opt_product, opt_template],
                [{"scheduler": sched, "interval": interval, "frequency": 1}],
            )
        return [opt_product, opt_template]

    def _build_lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Optional[Tuple[Any, str]]:
        """Build an LR scheduler for *optimizer* from config.

        Returns ``(scheduler, interval)`` or None when scheduling is disabled.
        """
        sched_cfg = self.cfg.get("scheduler", {})
        if not sched_cfg.get("enabled", False):
            return None

        name = sched_cfg.get("name", "cosine")
        if name == "warmup_cosine":
            warmup_steps = int(sched_cfg.get("warmup_steps", 1000))
            total_steps_cfg = sched_cfg.get("total_steps")
            total_steps = (
                int(self.trainer.estimated_stepping_batches)
                if total_steps_cfg is None
                else int(total_steps_cfg)
            )
            min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.01))

            def _lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
                return max(min_lr_ratio, cosine)

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda), "step"

        if name == "cosine":
            return (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=sched_cfg.get("t_max", self.cfg["training"]["epochs"]),
                    eta_min=sched_cfg.get("eta_min", 1.0e-6),
                ),
                "epoch",
            )
        if name == "step":
            return (
                torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=sched_cfg.get("step_size", 1),
                    gamma=sched_cfg.get("gamma", 0.5),
                ),
                "epoch",
            )
        return (
            torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=sched_cfg.get("gamma", 0.95)
            ),
            "epoch",
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Initialize EMA and trainable-scorer services after parent asset setup.

        Calls the parent hook (which loads the template library, builds embeddings,
        and initializes the retrieval index), then additionally:
        - Loads the token feature cache for fast live template encoding.
        - Initializes the EMA shadow encoder when EMA is enabled.
        - Creates the :class:`Stage2TrainableTemplateMixin` scoring service.
        """
        super().on_fit_start()

        # Load feature cache for trainable-template live encoding.
        self.feature_cache = self.asset_manager._load_feature_cache(self.template_list)

        # EMA encoder service.
        if self._ema_cfg.ema.enabled:
            self.ema_encoder = Stage2EMAMixin(
                model=self.model,
                template_input_builder=self.template_input_builder,
                template_input_collator=self.template_input_collator,
                emb_cfg=self._cfg.template_embeddings,
                decay=self._ema_cfg.ema.decay,
            )
            self.ema_encoder.initialize(self.device)
            if getattr(self.trainer, "is_global_zero", True):
                print(
                    f"[stage2_ema] EMA encoder initialized "
                    f"(decay={self._ema_cfg.ema.decay}, device={self.device})"
                )

        # Trainable-template scoring service (always created; only used when active).
        self.trainable_scorer = Stage2TrainableTemplateMixin(
            model=self.model,
            template_input_builder=self.template_input_builder,
            template_input_collator=self.template_input_collator,
            trainable_cfg=self._ema_cfg.trainable_template,
        )

    def on_train_epoch_start(self) -> None:
        """Synchronize freeze state, reset optimizer, log, and refresh assets.

        Overrides the parent hook to:
        1. Zero optimizer gradients (discard any leftover partial accumulation).
        2. Detect freeze-state transitions and reset template Adam state on unfreeze.
        3. Decide whether to rebuild template embeddings (always rebuild when EMA is
           enabled; conditional on freeze change when EMA is off).
        """
        for opt in (self._opt_product, self._opt_template):
            if opt is not None:
                opt.zero_grad()

        prev_template_frozen = self.freeze_scheduler._template_frozen
        template_frozen, product_frozen, changed = self.freeze_scheduler.apply_for_epoch(
            self.current_epoch
        )

        if changed and not template_frozen and prev_template_frozen is True:
            self._reset_template_optimizer()

        self._log_epoch_freeze_state(
            int(self.current_epoch), template_frozen, product_frozen
        )

        if self._should_refresh_assets(template_frozen, changed):
            self._refresh_assets(force_rebuild=True)

    def on_train_epoch_end(self) -> None:
        """Step epoch-interval LR schedulers at the end of each training epoch."""
        if self._sched_product is not None and self._sched_product_interval == "epoch":
            self._sched_product.step()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Execute one Stage-2 EMA training step.

        Routes to the multi-positive contrastive objective during template-trainable
        epochs when configured; otherwise uses the same three-term listwise objective
        as the base Stage-2 but with live TE scoring and manual optimization.

        Args:
            batch: Collated training batch dict.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            Scalar total loss (used by Lightning for logging; optimizer stepping is
            done manually inside this method).
        """
        product_inputs = self._extract_product_inputs(batch)
        pos_template_ids: List[List[int]] = batch["pos_template_ids"]
        appl_template_ids: List[List[int]] = batch.get(
            "appl_template_ids", [[] for _ in pos_template_ids]
        )
        batch_has_applicable = any(len(a) > 0 for a in appl_template_ids)

        _, product_emb = self.model.encode_product(product_inputs)
        norm_product_emb = F.normalize(product_emb, dim=-1)

        template_trainable = self._is_template_trainable()

        # ── Template-phase contrastive objective ─────────────────────────
        if self._ema_cfg.trainable_template.uses_multi_positive_contrastive() and template_trainable:
            total_loss, loss_p2t, loss_t2p = self._compute_template_contrastive_loss(
                norm_product_emb, pos_template_ids
            )
            ema_kl_loss = self._compute_ema_contrastive_kl_loss(
                norm_product_emb, pos_template_ids
            )
            total_loss = total_loss + (
                self._ema_cfg.trainable_template.ema_kl_weight * ema_kl_loss
            )
            total_loss, skipped_for_non_finite = self._guard_non_finite_loss(
                total_loss,
                batch_idx,
                objective_name="template_contrastive",
            )
            if skipped_for_non_finite:
                zero_loss = total_loss.new_zeros(())
                self.log("train_template_contrastive_loss", zero_loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log("train_template_contrastive_p2t", zero_loss, on_step=True, on_epoch=True, prog_bar=False)
                if loss_t2p is not None:
                    self.log("train_template_contrastive_t2p", zero_loss, on_step=True, on_epoch=True, prog_bar=False)
                self.log("train_l_applicable", zero_loss, on_step=True, on_epoch=True, prog_bar=False)
                self.log("train_l_penalty", zero_loss, on_step=True, on_epoch=True, prog_bar=False)
                self.log("train_l_ema_kl", zero_loss, on_step=True, on_epoch=True, prog_bar=False)
                self.log("train_loss", zero_loss, on_step=True, on_epoch=True, prog_bar=True)
                return zero_loss
            self._manual_backward_and_step(total_loss, batch_idx)
            self.log("train_template_contrastive_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_template_contrastive_p2t", loss_p2t, on_step=True, on_epoch=True, prog_bar=False)
            if loss_t2p is not None:
                self.log("train_template_contrastive_t2p", loss_t2p, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_l_applicable", total_loss.new_tensor(0.0), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_l_penalty", total_loss.new_tensor(0.0), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_l_ema_kl", ema_kl_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
            return total_loss

        # ── Standard listwise path ────────────────────────────────────────
        cand_ids, pos_mask = self._build_training_candidates(
            norm_product_emb, pos_template_ids, template_trainable
        )
        total_loss, listwise_loss, entropy, scores = self._compute_train_batch_loss(
            norm_product_emb, cand_ids, pos_mask, template_trainable
        )
        ema_kl_loss = self._compute_ema_candidate_kl_loss(
            norm_product_emb, cand_ids, scores
        )
        total_loss = total_loss + (
            self._ema_cfg.trainable_template.ema_kl_weight * ema_kl_loss
        )

        # Aux losses (GPU embeddings required for L_applicable / L_penalty)
        applicable_loss = total_loss.new_tensor(0.0)
        penalty_loss = total_loss.new_tensor(0.0)
        loss_cfg = self._cfg.loss
        if batch_has_applicable and self.template_emb_gpu is not None:
            template_emb = self.template_emb_gpu.to(dtype=norm_product_emb.dtype)
            scores_full = norm_product_emb @ template_emb.t()  # [B, N]
            allowed_negative_mask = (
                self._get_train_negative_mask(self.device)
                if self._cfg.candidate_sampling.restrict_negatives_to_train_templates
                else None
            )

            if loss_cfg.applicable_loss_weight > 0:
                appl_cand_ids, appl_pos_mask = self.candidate_sampler.build_applicable_candidates_gpu(
                    scores_full=scores_full,
                    appl_template_ids=appl_template_ids,
                    pos_template_ids=pos_template_ids,
                    sampling_cfg=self._cfg.candidate_sampling,
                    device=self.device,
                    allowed_negative_mask_full=allowed_negative_mask,
                )
                scores_appl = scores_full.gather(1, appl_cand_ids)
                if loss_cfg.temperature > 0:
                    scores_appl = scores_appl / loss_cfg.temperature
                applicable_loss, _, _ = self._compute_listwise_loss(scores_appl, appl_pos_mask)

            if loss_cfg.penalty_loss_weight > 0:
                penalty_loss = self._compute_penalty_loss(
                    scores_full,
                    appl_template_ids,
                    pos_template_ids,
                    k=loss_cfg.penalty_top_k,
                    allowed_negative_mask_full=allowed_negative_mask,
                )

            total_loss = (
                total_loss
                + loss_cfg.applicable_loss_weight * applicable_loss
                + loss_cfg.penalty_loss_weight * penalty_loss
            )

        total_loss, skipped_for_non_finite = self._guard_non_finite_loss(
            total_loss,
            batch_idx,
            objective_name="listwise",
        )
        if skipped_for_non_finite:
            zero_loss = total_loss.new_zeros(())
            self.log("train_listwise_loss", zero_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_entropy", zero_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_l_applicable", zero_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_l_penalty", zero_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_l_ema_kl", zero_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_loss", zero_loss, on_step=True, on_epoch=True, prog_bar=True)
            return zero_loss

        self._manual_backward_and_step(total_loss, batch_idx)
        self.log("train_listwise_loss", listwise_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_entropy", entropy, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_l_applicable", applicable_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_l_penalty", penalty_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_l_ema_kl", ema_kl_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    # ------------------------------------------------------------------
    # Candidate construction (phase-aware)
    # ------------------------------------------------------------------

    def _build_training_candidates(
        self,
        norm_product_emb: torch.Tensor,
        pos_template_ids: List[List[int]],
        template_trainable: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build training candidates using the phase-active sampling config.

        When the template encoder is trainable the GPU matrix builder is disabled
        by default (to avoid a dense B×N matmul that would require a GPU-resident
        cache) unless explicitly re-enabled in ``trainable_template.candidate_sampling``.

        Args:
            norm_product_emb: L2-normalized product embeddings ``[B, D]``.
            pos_template_ids: Positive template ID lists per sample.
            template_trainable: Whether TE is currently trainable.

        Returns:
            ``(candidate_ids [B, C], positive_mask [B, C])``.
        """
        sampling_cfg = self._resolve_active_train_sampling_cfg(template_trainable)

        if self._should_use_gpu_candidate_builder(template_trainable):
            return self.candidate_sampler.build_train_candidates_gpu(
                norm_product_emb=norm_product_emb,
                pos_template_ids=pos_template_ids,
                template_emb_gpu=self.template_emb_gpu,
                sampling_cfg=sampling_cfg,
                device=self.device,
                allowed_negative_mask_full=(
                    self._get_train_negative_mask(self.device)
                    if sampling_cfg.restrict_negatives_to_train_templates
                    else None
                ),
            )

        n_templates = self.template_emb_cpu.size(0)
        retrieval_rows = self._build_retrieval_rows(
            norm_product_emb, sampling_cfg, len(pos_template_ids)
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

    def _resolve_active_train_sampling_cfg(
        self, template_trainable: bool
    ) -> CandidateSamplingConfig:
        """Resolve the effective candidate sampling config for the current phase.

        Start from the base config in all phases, then optionally merge
        trainable-phase overrides.  Negative budgets are always clamped to fit
        candidate_size (old behavior), with an optional trainable-phase candidate
        cap applied before clamping.
        """
        base = self._cfg.candidate_sampling

        raw: Dict[str, Any] = {
            "candidate_size": base.candidate_size,
            "hard_negatives": base.hard_negatives,
            "inbatch_negatives": base.inbatch_negatives,
            "random_negatives": base.random_negatives,
            "restrict_negatives_to_train_templates": bool(
                base.restrict_negatives_to_train_templates
            ),
            "retrieval": dict(base.retrieval) if base.retrieval else {},
        }

        if template_trainable:
            tcfg = self._ema_cfg.trainable_template
            override = tcfg.candidate_sampling
            limit = tcfg.max_candidates_per_product

            for k, v in override.items():
                if k == "retrieval":
                    continue
                raw[k] = v
            retrieval_overrides = override.get("retrieval", {})
            if isinstance(retrieval_overrides, dict):
                raw["retrieval"].update(retrieval_overrides)

            if limit is not None and int(raw.get("candidate_size", base.candidate_size)) > limit:
                raw["candidate_size"] = limit

        cand_size = max(1, int(raw["candidate_size"]))
        raw["candidate_size"] = cand_size
        hard, inbatch, rand = Stage2CandidateSampler.fit_negative_budgets_to_candidate_size(
            cand_size,
            int(raw["hard_negatives"]),
            int(raw["inbatch_negatives"]),
            int(raw["random_negatives"]),
        )
        raw["hard_negatives"] = hard
        raw["inbatch_negatives"] = inbatch
        raw["random_negatives"] = rand
        return CandidateSamplingConfig.from_dict(raw)

    def _build_retrieval_rows(
        self,
        norm_product_emb: torch.Tensor,
        sampling_cfg: CandidateSamplingConfig,
        batch_size: int,
    ) -> List[List[int]]:
        """Build retrieval rows used for training hard-negative construction."""
        retrieval_cfg = sampling_cfg.retrieval
        rows: List[List[int]] = [[] for _ in range(batch_size)]
        if not retrieval_cfg.get("enabled", True) or sampling_cfg.hard_negatives <= 0:
            return rows
        top_k = max(
            int(retrieval_cfg.get("top_k", sampling_cfg.hard_negatives * 4)),
            sampling_cfg.hard_negatives,
        )
        allow_matrix = bool(retrieval_cfg.get("allow_matrix_in_training", False))
        return self.retrieval.retrieve(norm_product_emb, top_k, allow_matrix)

    def _should_use_gpu_candidate_builder(self, template_trainable: bool) -> bool:
        if self.template_emb_gpu is None:
            return False
        if not template_trainable:
            return True
        # Trainable phase defaults to CPU retrieval; opt-in via config.
        override = self._ema_cfg.trainable_template.candidate_sampling
        if isinstance(override, dict) and "use_gpu_candidate_builder" in override:
            return bool(override["use_gpu_candidate_builder"])
        return False

    # ------------------------------------------------------------------
    # Scoring (routes to trainable or frozen path)
    # ------------------------------------------------------------------

    def _score_candidate_templates(
        self,
        norm_product_emb: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates via live TE (trainable phase) or frozen cache.

        Returns temperature-scaled scores ready for loss computation.
        """
        temp = self._cfg.loss.temperature
        if self._is_template_trainable():
            raw_scores = self.trainable_scorer.score_candidates(
                norm_product_emb, candidate_ids, self.feature_cache, self.template_list, self.device
            )
        else:
            raw_scores = self._score_from_frozen_cache(norm_product_emb, candidate_ids)
        return raw_scores / temp if temp > 0 else raw_scores

    def _score_from_frozen_cache(
        self,
        norm_product_emb: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Gather frozen candidate embeddings and compute dot-product scores."""
        batch_size, cand_count = candidate_ids.shape
        if self.template_emb_gpu is not None:
            cand_emb = self.template_emb_gpu[candidate_ids]
        else:
            flat_idx = candidate_ids.view(-1).detach().cpu()
            uniq_idx, inverse = torch.unique(flat_idx, sorted=False, return_inverse=True)
            uniq_emb = self.template_emb_cpu.index_select(0, uniq_idx).to(self.device)
            cand_emb = uniq_emb[inverse].view(batch_size, cand_count, -1)
        cand_emb = cand_emb.to(dtype=norm_product_emb.dtype)
        return Stage2TrainableTemplateMixin.score_from_embeddings(norm_product_emb, cand_emb)

    # ------------------------------------------------------------------
    # Train batch loss (with micro-batching for trainable phase)
    # ------------------------------------------------------------------

    def _compute_train_batch_loss(
        self,
        norm_product_emb: torch.Tensor,
        cand_ids: torch.Tensor,
        pos_mask: torch.Tensor,
        template_trainable: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute train loss, with micro-batching when template is trainable.

        When micro-batching is active, all unique candidate templates are encoded
        once globally (deduplication across micro-batches avoids redundant TE
        forward passes) and the loss is accumulated as a weighted sum.
        """
        micro_bs = self._ema_cfg.trainable_template.micro_batch_size
        batch_size = norm_product_emb.size(0)
        use_chunking = (
            template_trainable
            and micro_bs is not None
            and micro_bs < batch_size
        )

        if not use_chunking:
            scores = self._score_candidate_templates(norm_product_emb, cand_ids)
            total, listwise, entropy = self._compute_listwise_loss(scores, pos_mask)
            return total, listwise, entropy, scores

        # Global unique dedup across micro-batches.
        cand_count = cand_ids.size(1)
        all_flat = cand_ids.reshape(-1).detach().cpu()
        global_uniq, global_inv = torch.unique(all_flat, sorted=False, return_inverse=True)
        global_uniq_emb = self.trainable_scorer.encode_unique_templates(
            global_uniq, self.feature_cache, self.template_list, self.device
        )
        global_inv = global_inv.to(device=self.device)
        global_cand_emb = global_uniq_emb[global_inv].view(batch_size, cand_count, -1)

        total = norm_product_emb.new_tensor(0.0)
        listwise = norm_product_emb.new_tensor(0.0)
        entropy_acc = norm_product_emb.new_tensor(0.0)
        score_chunks: List[torch.Tensor] = []
        temp = self._cfg.loss.temperature

        for start in range(0, batch_size, micro_bs):
            end = min(start + micro_bs, batch_size)
            chunk_scores = Stage2TrainableTemplateMixin.score_from_embeddings(
                norm_product_emb[start:end], global_cand_emb[start:end]
            )
            if temp > 0:
                chunk_scores = chunk_scores / temp
            ct, cl, ce = self._compute_listwise_loss(chunk_scores, pos_mask[start:end])
            weight = float(end - start) / float(batch_size)
            total = total + ct * weight
            listwise = listwise + cl * weight
            entropy_acc = entropy_acc + ce * weight
            score_chunks.append(chunk_scores)

        return total, listwise, entropy_acc, torch.cat(score_chunks, dim=0)

    # ------------------------------------------------------------------
    # Template-phase contrastive loss
    # ------------------------------------------------------------------

    def _compute_template_contrastive_loss(
        self,
        norm_product_emb: torch.Tensor,
        pos_template_ids: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute multi-positive contrastive loss over the batch's positive templates.

        Only the unique positive templates present in this batch are encoded, so
        the number of TE forward passes is bounded by the batch's unique positive set
        rather than the full candidate size.

        Args:
            norm_product_emb: L2-normalized product embeddings ``[B, D]``.
            pos_template_ids: Positive template ID lists per sample.

        Returns:
            ``(total_loss, loss_p2t, loss_t2p)`` — loss_t2p is None when
            ``symmetric_contrastive=False``.
        """
        n_templates = len(self.template_list)
        normalized_pos = [
            Stage2CandidateSampler.normalize_positive_template_ids(
                row, n_templates, None, "Stage-2 EMA contrastive"
            )
            for row in pos_template_ids
        ]

        uniq_sorted = sorted({tid for row in normalized_pos for tid in row})
        if not uniq_sorted:
            raise RuntimeError(
                "Stage-2 EMA contrastive requires at least one valid positive template ID."
            )

        uniq_ids = torch.tensor(uniq_sorted, dtype=torch.long)
        template_inputs = self.trainable_scorer.build_inputs(
            uniq_ids, self.feature_cache, self.template_list, self.device
        )
        template_emb = self.trainable_scorer._encode_in_chunks(template_inputs)
        template_emb = F.normalize(template_emb, dim=-1)

        prod_emb = F.normalize(norm_product_emb, dim=-1)
        tcfg = self._ema_cfg.trainable_template
        temp = (
            tcfg.contrastive_temperature
            if tcfg.contrastive_temperature is not None
            else self._cfg.loss.temperature
        )
        logits = prod_emb @ template_emb.t()
        if temp > 0:
            logits = logits / temp

        col_map = {tid: idx for idx, tid in enumerate(uniq_sorted)}
        B, C = len(normalized_pos), len(uniq_sorted)
        pos_mask_p2t = torch.zeros(B, C, dtype=torch.bool, device=self.device)
        for row_idx, row_pos in enumerate(normalized_pos):
            cols = torch.tensor(
                [col_map[tid] for tid in row_pos], dtype=torch.long, device=self.device
            )
            pos_mask_p2t[row_idx, cols] = True

        loss_p2t = self._multi_positive_cross_entropy(logits, pos_mask_p2t)

        if not tcfg.symmetric_contrastive:
            return loss_p2t, loss_p2t, None

        loss_t2p = self._multi_positive_cross_entropy(
            logits.t(), pos_mask_p2t.t().contiguous()
        )
        return 0.5 * (loss_p2t + loss_t2p), loss_p2t, loss_t2p

    @staticmethod
    def _multi_positive_cross_entropy(
        logits: torch.Tensor, positive_mask: torch.Tensor
    ) -> torch.Tensor:
        """Multi-positive NLL: mean NLL over rows that have at least one positive."""
        log_probs = F.log_softmax(logits, dim=-1)
        pos_counts = positive_mask.sum(dim=-1)
        valid = pos_counts > 0
        if not bool(valid.any()):
            raise RuntimeError(
                "Multi-positive contrastive received rows with no positive templates."
            )
        masked = torch.where(positive_mask, log_probs, torch.zeros_like(log_probs))
        per_row = -(masked.sum(dim=-1) / pos_counts.clamp_min(1).float())
        return per_row[valid].mean()

    # ------------------------------------------------------------------
    # EMA-teacher KL regularization
    # ------------------------------------------------------------------

    def _compute_ema_candidate_kl_loss(
        self,
        norm_product_emb: torch.Tensor,
        candidate_ids: torch.Tensor,
        student_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Distill the live scorer toward the EMA teacher over sampled candidates."""
        if not self._should_apply_ema_kl():
            return student_scores.new_tensor(0.0)

        teacher_scores = self.ema_encoder.score_candidates(
            norm_product_emb,
            candidate_ids,
            self.template_list,
            self.device,
            self.feature_cache,
        )
        temp = self._cfg.loss.temperature
        if temp > 0:
            teacher_scores = teacher_scores / temp
        return self._compute_ema_kl_divergence(student_scores, teacher_scores)

    def _compute_ema_contrastive_kl_loss(
        self,
        norm_product_emb: torch.Tensor,
        pos_template_ids: List[List[int]],
    ) -> torch.Tensor:
        """Distill live TE toward EMA TE over the batch's unique positive templates."""
        if not self._should_apply_ema_kl():
            return norm_product_emb.new_tensor(0.0)

        n_templates = len(self.template_list)
        normalized_pos = [
            Stage2CandidateSampler.normalize_positive_template_ids(
                row, n_templates, None, "Stage-2 EMA KL"
            )
            for row in pos_template_ids
        ]
        uniq_sorted = sorted({tid for row in normalized_pos for tid in row})
        if not uniq_sorted:
            return norm_product_emb.new_tensor(0.0)

        uniq_ids = torch.tensor(uniq_sorted, dtype=torch.long)
        student_inputs = self.trainable_scorer.build_inputs(
            uniq_ids, self.feature_cache, self.template_list, self.device
        )
        student_emb = self.trainable_scorer._encode_in_chunks(student_inputs)
        student_emb = F.normalize(student_emb, dim=-1)

        teacher_emb = self.ema_encoder.encode_unique_templates(
            uniq_ids, self.template_list, self.device, self.feature_cache
        ).to(device=self.device, dtype=student_emb.dtype)

        prod_emb = F.normalize(norm_product_emb, dim=-1)
        tcfg = self._ema_cfg.trainable_template
        temp = (
            tcfg.contrastive_temperature
            if tcfg.contrastive_temperature is not None
            else self._cfg.loss.temperature
        )
        student_logits = prod_emb @ student_emb.t()
        teacher_logits = prod_emb @ teacher_emb.t()
        if temp > 0:
            student_logits = student_logits / temp
            teacher_logits = teacher_logits / temp
        return self._compute_ema_kl_divergence(student_logits, teacher_logits)

    def _should_apply_ema_kl(self) -> bool:
        return (
            self._is_template_trainable()
            and self.ema_encoder is not None
            and self.trainable_scorer is not None
            and self._ema_cfg.trainable_template.ema_kl_weight > 0.0
        )

    @staticmethod
    def _compute_ema_kl_divergence(
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Forward-KL teacher distillation over candidate logits."""
        teacher_probs = F.softmax(teacher_scores.detach(), dim=-1)
        student_log_probs = F.log_softmax(student_scores, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    # ------------------------------------------------------------------
    # Manual optimization helpers
    # ------------------------------------------------------------------

    def _guard_non_finite_loss(
        self,
        loss: torch.Tensor,
        batch_idx: int,
        objective_name: str,
    ) -> Tuple[torch.Tensor, bool]:
        """Optionally skip the current batch when loss is NaN/Inf.

        Args:
            loss: Scalar training loss to validate.
            batch_idx: Current batch index within the epoch.
            objective_name: Human-readable label for diagnostics.

        Returns:
            ``(loss_or_zero, skipped)`` where ``skipped`` is True only when
            guard is enabled and the incoming loss is non-finite.
        """
        if not self._nan_guard_enabled:
            return loss, False

        if bool(torch.isfinite(loss).all()):
            return loss, False

        self._clear_all_optimizer_gradients()
        self._non_finite_loss_skip_count += 1
        if getattr(self.trainer, "is_global_zero", True):
            print(
                "[stage2_ema.nan_guard] skipped batch due to non-finite loss "
                f"(objective={objective_name}, epoch={self.current_epoch}, batch={batch_idx}, "
                f"total_skipped={self._non_finite_loss_skip_count})"
            )
        self.log("train_nan_guard_skipped_loss", loss.new_tensor(1.0), on_step=True, on_epoch=True, prog_bar=False)
        return loss.new_zeros(()), True

    def _should_skip_for_non_finite_gradients(self, batch_idx: int) -> bool:
        """Return True when non-finite gradients were detected and cleared."""
        if not self._nan_guard_enabled:
            return False
        if not self._has_non_finite_gradients():
            return False

        self._clear_all_optimizer_gradients()
        self._non_finite_grad_skip_count += 1
        if getattr(self.trainer, "is_global_zero", True):
            print(
                "[stage2_ema.nan_guard] skipped optimizer step due to non-finite gradients "
                f"(epoch={self.current_epoch}, batch={batch_idx}, "
                f"total_skipped={self._non_finite_grad_skip_count})"
            )
        self.log("train_nan_guard_skipped_grad", 1.0, on_step=True, on_epoch=True, prog_bar=False)
        return True

    def _has_non_finite_gradients(self) -> bool:
        """Return True if any active optimizer currently holds NaN/Inf gradients."""
        active_optimizers: List[torch.optim.Optimizer] = []
        if self._opt_product is not None and not self.freeze_scheduler._product_frozen:
            active_optimizers.append(self._opt_product)
        if self._opt_template is not None and self._is_template_trainable():
            active_optimizers.append(self._opt_template)

        for optimizer in active_optimizers:
            if self._optimizer_has_non_finite_gradients(optimizer):
                return True
        return False

    @staticmethod
    def _optimizer_has_non_finite_gradients(optimizer: torch.optim.Optimizer) -> bool:
        """Return True if *optimizer* has any NaN/Inf gradient tensor."""
        for group in optimizer.param_groups:
            params = group.get("params", [])
            for param in params:
                grad = getattr(param, "grad", None)
                if not isinstance(grad, torch.Tensor):
                    continue
                grad_values = grad.coalesce().values() if grad.is_sparse else grad
                if not bool(torch.isfinite(grad_values).all()):
                    return True
        return False

    def _clear_all_optimizer_gradients(self) -> None:
        """Clear gradients on both optimizers, handling older zero_grad signatures."""
        for optimizer in (self._opt_product, self._opt_template):
            if optimizer is None:
                continue
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                optimizer.zero_grad()

    def _manual_backward_and_step(
        self, loss: torch.Tensor, batch_idx: int
    ) -> None:
        """Accumulate gradients and step only the phase-active optimizer(s).

        Loss is scaled by ``1/accum`` so effective gradient magnitude is
        independent of accumulation depth.  Optimizer stepping and scheduler
        stepping happen only at the accumulation boundary.

        Args:
            loss: Scalar loss tensor.
            batch_idx: Current batch index within the epoch.
        """
        accum = self._resolve_accumulation_depth()
        clip_val = float(self.cfg["training"].get("max_grad_norm", 1.0))

        self.manual_backward(loss / accum)
        if self._should_skip_for_non_finite_gradients(batch_idx):
            return

        if (batch_idx + 1) % accum != 0:
            return

        if self._opt_product is not None:
            if not self.freeze_scheduler._product_frozen:
                if clip_val > 0:
                    self.clip_gradients(self._opt_product, gradient_clip_val=clip_val)
                self._opt_product.step()
                if (
                    self._sched_product is not None
                    and self._sched_product_interval == "step"
                ):
                    self._sched_product.step()
            self._opt_product.zero_grad()

        if self._opt_template is not None:
            if self._is_template_trainable():
                if clip_val > 0:
                    self.clip_gradients(self._opt_template, gradient_clip_val=clip_val)
                self._opt_template.step()
                if self._sched_template is not None:
                    self._sched_template.step()
                if self.ema_encoder is not None:
                    self.ema_encoder.update_from_live()
            self._opt_template.zero_grad()

    def _reset_template_optimizer(self) -> None:
        """Clear template Adam state on entry to a newly trainable phase.

        Prevents stale near-zero second moments (accumulated when TE was frozen
        and received zero gradients) from producing incorrect LR scaling on the
        first real update after unfreezing.
        """
        if self._opt_template is None:
            return
        if not self._ema_cfg.reset_template_optimizer_on_unfreeze:
            return
        self._opt_template.state.clear()
        if getattr(self.trainer, "is_global_zero", True):
            print(
                f"[stage2_ema] Reset template optimizer Adam state at epoch "
                f"{self.current_epoch} (frozen → trainable transition)."
            )

    def _resolve_accumulation_depth(self) -> int:
        """Return gradient accumulation depth for the current phase."""
        base = max(1, int(self.cfg["training"].get("accumulate_grad_batches", 1)))
        if not self._is_template_trainable():
            return base
        phase_accum = self._ema_cfg.trainable_template.accumulate_grad_batches
        return phase_accum if phase_accum is not None else base

    # ------------------------------------------------------------------
    # Asset refresh (EMA-aware)
    # ------------------------------------------------------------------

    def _should_refresh_assets(
        self, template_frozen: bool, freeze_state_changed: bool
    ) -> bool:
        """Return whether the template embedding cache must be rebuilt this epoch.

        When EMA is enabled the cache is always stale at epoch start (the EMA
        encoder updated every step throughout the previous epoch).  Without EMA,
        refresh only on alternating-schedule transitions.
        """
        if self._ema_cfg.ema.enabled:
            return True
        if not self._ema_cfg.freeze.alternate:
            return False
        return freeze_state_changed or not template_frozen

    def _refresh_assets(self, force_rebuild: bool) -> None:
        """Rebuild the template embedding cache and retrieval index.

        Uses EMA-TE when available (smoother, more stable weights); falls back
        to live TE otherwise.
        """
        if self.ema_encoder is not None:
            self.template_emb_cpu = self.ema_encoder.encode_all_templates(
                self.template_list, self.device, self.feature_cache
            )
        else:
            self.template_emb_cpu = self.asset_manager.load_or_build_embeddings(
                self.template_list, self.device, force_rebuild=force_rebuild
            )

        self.template_emb_gpu = (
            self.template_emb_cpu.to(self.device)
            if self._cfg.keep_embeddings_on_gpu
            else None
        )
        self.retrieval.initialize(self.template_emb_cpu)

    # ------------------------------------------------------------------
    # Phase helpers and logging
    # ------------------------------------------------------------------

    def _is_template_trainable(self) -> bool:
        """Return True when the template encoder currently has trainable parameters."""
        return any(p.requires_grad for p in self.model.template_encoder.parameters())

    def _log_epoch_freeze_state(
        self, epoch: int, template_frozen: bool, product_frozen: bool
    ) -> None:
        scoring_mode = (
            "live_template_encoding"
            if self._is_template_trainable()
            else "cached_template_embeddings"
        )
        if getattr(self.trainer, "is_global_zero", True):
            print(
                f"[stage2_ema.freeze] epoch={epoch} "
                f"template_frozen={template_frozen} product_frozen={product_frozen} "
                f"scoring_mode={scoring_mode}"
            )
        self.log("stage2_template_frozen", float(template_frozen), on_step=False, on_epoch=True)
        self.log("stage2_product_frozen", float(product_frozen), on_step=False, on_epoch=True)


# Backward-compatible alias (the top-level __init__ imports this name).
Stage2LightningModule = Stage2EmaLightningModule
