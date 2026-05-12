"""Type aliases and configuration dataclasses for the stage_two_ema package.

Re-exports everything from stage_two.datatypes and adds the EMA-specific
additions: EMAConfig, TrainableTemplateConfig, EmaFreezeConfig, Stage2EmaConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Re-export shared types so callers can import from one place.
from ..stage_two.datatypes import (  # noqa: F401
    CandidateRow,
    CandidateSamplingConfig,
    EvaluationConfig,
    FeatureCache,
    FreezeConfig,
    LossConfig,
    PositiveMaskRow,
    Stage2Config,
    TemplateCacheConfig,
    TemplateEmbeddingConfig,
)


# ---------------------------------------------------------------------------
# EMA configuration
# ---------------------------------------------------------------------------


@dataclass
class EMAConfig:
    """EMA template-encoder stabilization settings.

    Attributes:
        enabled: Whether EMA tracking is active.  When False all EMA code paths
            are skipped and the module behaves like plain Stage-2.
        decay: EMA momentum coefficient (``ema = decay * ema + (1-decay) * live``).
            Typical values: 0.99 – 0.9999.
    """

    enabled: bool
    decay: float

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> EMAConfig:
        """Parse from ``cfg["training"]["stage2"]["ema"]``."""
        return cls(
            enabled=bool(raw.get("enabled", False)),
            decay=float(raw.get("decay", 0.999)),
        )


# ---------------------------------------------------------------------------
# Trainable-template phase configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainableTemplateConfig:
    """Hyper-parameters governing the trainable-template phase of Stage-2 EMA.

    Attributes:
        objective: Loss objective in the trainable phase.  ``"listwise"`` (default)
            uses the same multi-positive listwise NLL as the frozen phase.
            ``"multi_positive_contrastive"`` uses a symmetric contrastive loss
            over only the batch's positive templates.
        contrastive_temperature: Override temperature for the contrastive objective
            (None = use global loss temperature).
        symmetric_contrastive: Whether to average P→T and T→P cross-entropy terms.
        max_candidates_per_product: Hard cap on candidate set size during trainable
            phase (to bound activation memory).  None = no cap beyond candidate_size.
        encode_batch_size: Chunk size for on-the-fly template encoding.
        micro_batch_size: Micro-batch size for the chunked loss loop.  None = full batch.
        train_last_k_layers: If set, only the last *k* transformer layers of the
            template encoder are unfrozen.  None = unfreeze all layers.
        train_template_projector: Whether the template projector trains in this phase.
            None = inherit from global freeze policy.
        train_template_embeddings: Whether token/position embeddings also train when
            using last-k-layer partial unfreeze.
        freeze_projector_with_template_encoder: When True, the projector is frozen
            together with the template encoder in fixed-frozen mode.
        gradient_checkpointing: Enable gradient checkpointing on the template encoder
            to reduce activation memory in the trainable phase.
        candidate_sampling: Per-phase overrides for the candidate sampling config dict
            (merged on top of the base ``stage2.candidate_sampling`` block).
        accumulate_grad_batches: Per-phase gradient accumulation depth.  None = use
            global ``training.accumulate_grad_batches``.
        ema_kl_weight: Weight of EMA-teacher KL regularization in template-trainable
            phases.  ``0.0`` disables the regularizer.
    """

    objective: str
    contrastive_temperature: Optional[float]
    symmetric_contrastive: bool
    max_candidates_per_product: Optional[int]
    encode_batch_size: int
    micro_batch_size: Optional[int]
    train_last_k_layers: Optional[int]
    train_template_projector: Optional[bool]
    train_template_embeddings: bool
    freeze_projector_with_template_encoder: bool
    gradient_checkpointing: bool
    candidate_sampling: Dict[str, Any]
    accumulate_grad_batches: Optional[int]
    ema_kl_weight: float

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> TrainableTemplateConfig:
        """Parse from ``cfg["training"]["stage2"]["trainable_template"]``."""
        contrastive_cfg = raw.get("multi_positive_contrastive", {})
        if not isinstance(contrastive_cfg, dict):
            contrastive_cfg = {}

        contrastive_temp: Optional[float] = None
        if contrastive_cfg.get("temperature") is not None:
            contrastive_temp = float(contrastive_cfg["temperature"])

        symmetric = bool(contrastive_cfg.get("symmetric", True))

        # Candidate limit: explicit key or legacy trainable_candidate_size fallback.
        max_cands_raw = raw.get("max_candidates_per_product")
        if max_cands_raw is None:
            phase_cs = raw.get("candidate_sampling", {})
            if isinstance(phase_cs, dict):
                max_cands_raw = phase_cs.get("trainable_candidate_size")
        max_cands: Optional[int] = None
        if max_cands_raw is not None:
            n = int(max_cands_raw)
            max_cands = n if n > 0 else None

        micro_bs_raw = raw.get("micro_batch_size")
        micro_bs: Optional[int] = None
        if micro_bs_raw is not None:
            n = int(micro_bs_raw)
            micro_bs = n if n > 0 else None

        last_k_raw = raw.get("train_last_k_layers")
        last_k: Optional[int] = None
        if last_k_raw is not None:
            n = int(last_k_raw)
            last_k = n if n > 0 else None

        projector_raw = raw.get("train_template_projector")
        projector: Optional[bool] = bool(projector_raw) if projector_raw is not None else None

        accum_raw = raw.get("accumulate_grad_batches")
        accum: Optional[int] = None
        if accum_raw is not None:
            n = int(accum_raw)
            accum = n if n >= 1 else None

        phase_cs = raw.get("candidate_sampling", {})
        candidate_sampling = dict(phase_cs) if isinstance(phase_cs, dict) else {}

        return cls(
            objective=str(raw.get("objective", "listwise")).strip().lower(),
            contrastive_temperature=contrastive_temp,
            symmetric_contrastive=symmetric,
            max_candidates_per_product=max_cands,
            encode_batch_size=max(1, int(raw.get("encode_batch_size", 128))),
            micro_batch_size=micro_bs,
            train_last_k_layers=last_k,
            train_template_projector=projector,
            train_template_embeddings=bool(raw.get("train_template_embeddings", False)),
            freeze_projector_with_template_encoder=bool(
                raw.get("freeze_projector_with_template_encoder", False)
            ),
            gradient_checkpointing=bool(raw.get("gradient_checkpointing", False)),
            candidate_sampling=candidate_sampling,
            accumulate_grad_batches=accum,
            ema_kl_weight=float(raw.get("ema_kl_weight", 0.0)),
        )

    def uses_multi_positive_contrastive(self) -> bool:
        """Return True when the trainable-phase objective is multi-positive contrastive."""
        return self.objective in {
            "multi_positive_contrastive",
            "multipositive_contrastive",
            "contrastive",
        }


# ---------------------------------------------------------------------------
# Extended freeze configuration (adds per-phase product encoder control)
# ---------------------------------------------------------------------------


@dataclass
class EmaFreezeConfig:
    """Freeze schedule with independent product-encoder control per phase.

    Extends the base :class:`~stage_two.datatypes.FreezeConfig` with two
    additional booleans that let the product encoder be frozen or unfrozen
    independently from the template encoder during each alternating phase.

    Attributes:
        mode: ``"fixed"`` or ``"alternate"``.
        template_frozen: Template-encoder freeze flag in fixed mode.
        product_frozen: Product-encoder freeze flag in fixed mode.
        alternate: Whether to alternate template freeze/unfreeze each epoch window.
        template_frozen_epochs: Consecutive frozen epochs (alternate mode).
        template_unfrozen_epochs: Consecutive trainable epochs (alternate mode).
        start_with_template_frozen: Phase at epoch 0 (alternate mode).
        start_epoch: Epoch at which the alternating schedule begins.
        product_frozen_while_template_frozen: Whether to freeze the product encoder
            during template-frozen epochs (alternate mode only).
        product_frozen_while_template_unfrozen: Whether to freeze the product encoder
            during template-trainable epochs (alternate mode only).
    """

    mode: str
    template_frozen: bool
    product_frozen: bool
    alternate: bool
    template_frozen_epochs: int
    template_unfrozen_epochs: int
    start_with_template_frozen: bool
    start_epoch: int
    product_frozen_while_template_frozen: bool
    product_frozen_while_template_unfrozen: bool

    @classmethod
    def fixed(
        cls,
        *,
        template_frozen: bool,
        product_frozen: bool,
    ) -> EmaFreezeConfig:
        """Build a fixed (non-alternating) freeze config."""
        return cls(
            mode="fixed",
            template_frozen=template_frozen,
            product_frozen=product_frozen,
            alternate=False,
            template_frozen_epochs=1,
            template_unfrozen_epochs=1,
            start_with_template_frozen=True,
            start_epoch=0,
            product_frozen_while_template_frozen=False,
            product_frozen_while_template_unfrozen=True,
        )

    @classmethod
    def from_dict(
        cls, raw: Any, *, default_product_frozen: bool
    ) -> EmaFreezeConfig:
        """Parse from ``cfg["training"]["stage2"]["freeze_template_encoder"]``."""
        if isinstance(raw, bool):
            return cls.fixed(template_frozen=raw, product_frozen=default_product_frozen)

        if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
            return cls.fixed(template_frozen=False, product_frozen=default_product_frozen)

        if bool(raw.get("alternate", False)):
            frozen_product = bool(raw.get("product_frozen_while_template_frozen", False))
            unfrozen_product = bool(raw.get("product_frozen_while_template_unfrozen", True))
            return cls(
                mode="alternate",
                template_frozen=False,
                product_frozen=False,
                alternate=True,
                template_frozen_epochs=max(1, int(raw.get("template_frozen_epochs", 1))),
                template_unfrozen_epochs=max(1, int(raw.get("template_unfrozen_epochs", 1))),
                start_with_template_frozen=bool(raw.get("start_with_template_frozen", True)),
                start_epoch=max(0, int(raw.get("start_epoch", 0))),
                product_frozen_while_template_frozen=frozen_product,
                product_frozen_while_template_unfrozen=unfrozen_product,
            )

        template_frozen = bool(raw.get("template_frozen", raw.get("enabled", False)))
        product_frozen = bool(raw.get("product_frozen", default_product_frozen))
        return cls.fixed(template_frozen=template_frozen, product_frozen=product_frozen)


# ---------------------------------------------------------------------------
# Composite EMA-stage config (additions on top of Stage2Config)
# ---------------------------------------------------------------------------


@dataclass
class Stage2EmaConfig:
    """EMA-specific additions parsed alongside the base :class:`Stage2Config`.

    The base Stage-2 settings (loss, candidate sampling, evaluation, etc.) are
    held in a separate :class:`Stage2Config` instance stored as ``self._cfg``
    on the module; this dataclass carries only what is new.

    Attributes:
        ema: EMA encoder settings.
        trainable_template: Trainable-template phase settings.
        freeze: Extended freeze schedule with per-phase product control.
        template_lr: Learning rate for the template optimizer (AdamW).
        template_warmup_steps: Linear warm-up steps for the template optimizer.
            0 disables warm-up (default, backwards-compatible). Useful when
            transferring a PE trained on a different dataset: the PE produces
            peaked score distributions that cause FP16 softmax overflow if the
            TE optimizer starts at full LR from step 0. The scheduler only
            advances during TE-trainable steps, so ALT frozen phases do not
            consume warm-up budget.
        reset_template_optimizer_on_unfreeze: Clear template Adam state when
            transitioning from frozen to trainable, to prevent stale moment
            estimates from corrupting the first real update.
        skip_non_finite_batches: Whether training should skip optimizer updates
            when a batch produces non-finite loss or gradients.
    """

    ema: EMAConfig
    trainable_template: TrainableTemplateConfig
    freeze: EmaFreezeConfig
    template_lr: float
    template_warmup_steps: int
    reset_template_optimizer_on_unfreeze: bool
    skip_non_finite_batches: bool

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any], base_lr: float) -> Stage2EmaConfig:
        """Parse EMA-specific settings from the full experiment config.

        Args:
            cfg: Full experiment config dict.
            base_lr: Product optimizer LR used as fallback for ``template_lr``
                (defaults to ``base_lr * 0.1`` when not explicitly set).
        """
        raw = cfg.get("training", {}).get("stage2", {})
        nan_guard_raw = raw.get("nan_guard", {})
        nan_guard_enabled = bool(raw.get("skip_non_finite_batches", False))
        if hasattr(nan_guard_raw, "get"):
            nan_guard_enabled = bool(nan_guard_raw.get("enabled", nan_guard_enabled))
        return cls(
            ema=EMAConfig.from_dict(raw.get("ema", {})),
            trainable_template=TrainableTemplateConfig.from_dict(
                raw.get("trainable_template", {})
            ),
            freeze=EmaFreezeConfig.from_dict(
                raw.get("freeze_template_encoder", True),
                default_product_frozen=bool(raw.get("freeze_product_encoder", False)),
            ),
            template_lr=float(raw.get("template_lr", base_lr * 0.1)),
            template_warmup_steps=int(raw.get("template_warmup_steps", 0)),
            reset_template_optimizer_on_unfreeze=bool(
                raw.get("reset_template_optimizer_on_unfreeze", True)
            ),
            skip_non_finite_batches=nan_guard_enabled,
        )
