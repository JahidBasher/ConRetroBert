"""Type aliases and parsed configuration dataclasses for the stage_two package.

All primitive type aliases (previously scattered across individual files) and
all config dataclasses (replacing raw ``Dict[str, Any]`` config drilling) live
here.  Every component in this package imports its types from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Primitive type aliases
# ---------------------------------------------------------------------------

# Dict of pre-tokenised tensor features keyed by field name (e.g. "input_ids").
FeatureCache = Dict[str, torch.Tensor]

# A single row of candidate template integer IDs.
CandidateRow = List[int]

# Boolean mask aligned with a CandidateRow (True = positive template).
PositiveMaskRow = List[bool]


# ---------------------------------------------------------------------------
# Configuration dataclasses — parsed once, used everywhere
# ---------------------------------------------------------------------------

@dataclass
class LossConfig:
    """Parsed Stage-2 loss hyper-parameters.

    Attributes:
        temperature: Softmax temperature applied to candidate scores.
        applicable_loss_weight: Weight λ1 on the L_applicable term.
        penalty_loss_weight: Weight λ2 on the L_penalty term.
        penalty_top_k: Number of top retrieved templates checked for
            non-applicability in L_penalty.
        label_smoothing: Label-smoothing factor for the listwise NLL.
        entropy_bonus: Coefficient subtracted from the listwise loss to
            encourage a more uniform distribution over negatives.
    """

    temperature: float
    applicable_loss_weight: float
    penalty_loss_weight: float
    penalty_top_k: int
    label_smoothing: float
    entropy_bonus: float

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], fallback_temperature: float) -> LossConfig:
        """Parse from ``cfg["training"]["stage2"]["loss"]``.

        Args:
            raw: The ``stage2.loss`` sub-dict (may be empty).
            fallback_temperature: Global training temperature used when
                ``loss.temperature`` is not explicitly set.

        Returns:
            Fully populated :class:`LossConfig` instance.
        """
        return cls(
            temperature=float(raw.get("temperature", fallback_temperature)),
            applicable_loss_weight=float(raw.get("lambda_applicable", 0.0)),
            penalty_loss_weight=float(raw.get("lambda_penalty", 0.0)),
            penalty_top_k=int(raw.get("penalty_top_k", 512)),
            label_smoothing=float(raw.get("label_smoothing", 0.0)),
            entropy_bonus=float(raw.get("entropy_bonus", 0.0)),
        )


@dataclass
class CandidateSamplingConfig:
    """Parsed Stage-2 candidate sampling hyper-parameters.

    Attributes:
        candidate_size: Total number of candidates per training sample.
        hard_negatives: Maximum hard negatives drawn from the retrieval index.
        inbatch_negatives: Maximum in-batch negatives drawn from other samples.
        random_negatives: Maximum randomly sampled negatives.
        restrict_negatives_to_train_templates: When True, train-time negatives
            are restricted to templates observed in the training split. Validation
            still uses the full template library.
        retrieval: Backend-specific retrieval settings kept as a raw dict
            (backend name, FAISS paths, nprobe, etc.).
    """

    candidate_size: int
    hard_negatives: int
    inbatch_negatives: int
    random_negatives: int
    restrict_negatives_to_train_templates: bool
    retrieval: Dict[str, Any]

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> CandidateSamplingConfig:
        """Parse from ``cfg["training"]["stage2"]["candidate_sampling"]``.

        Args:
            raw: The ``stage2.candidate_sampling`` sub-dict (may be empty).

        Returns:
            Fully populated :class:`CandidateSamplingConfig` instance.
        """
        return cls(
            candidate_size=int(raw.get("candidate_size", 1024)),
            hard_negatives=int(raw.get("hard_negatives", 768)),
            inbatch_negatives=int(raw.get("inbatch_negatives", 192)),
            random_negatives=int(raw.get("random_negatives", 64)),
            restrict_negatives_to_train_templates=bool(
                raw.get("restrict_negatives_to_train_templates", False)
            ),
            retrieval=raw.get("retrieval", {}),
        )


@dataclass
class EvaluationConfig:
    """Parsed Stage-2 evaluation settings.

    Attributes:
        candidate_size: Number of candidates scored during validation.
        retrieval_top_k: Number of templates retrieved for the CPU candidate pool.
        compute_topk_metrics: Whether to compute top-k accuracy/recall metrics.
        top_k: List of k values for top-k reporting.
    """

    candidate_size: int
    retrieval_top_k: int
    compute_topk_metrics: bool
    top_k: List[int]

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], default_candidate_size: int) -> EvaluationConfig:
        """Parse from ``cfg["training"]["stage2"]["evaluation"]``.

        Args:
            raw: The ``stage2.evaluation`` sub-dict (may be empty).
            default_candidate_size: Fallback candidate size from
                :class:`CandidateSamplingConfig`.

        Returns:
            Fully populated :class:`EvaluationConfig` instance.
        """
        candidate_size = int(raw.get("candidate_size", default_candidate_size))
        return cls(
            candidate_size=candidate_size,
            retrieval_top_k=int(raw.get("retrieval_top_k", max(candidate_size, 2048))),
            compute_topk_metrics=bool(raw.get("compute_topk_metrics", True)),
            top_k=[int(k) for k in raw.get("top_k", [1, 3, 5, 10, 20, 50])],
        )


@dataclass
class TemplateEmbeddingConfig:
    """Parsed template embedding cache settings.

    Attributes:
        load_path: Path to load prebuilt embeddings from (None = always build).
        save_path: Path to save newly built embeddings (None = do not save).
        encode_batch_size: Batch size used when encoding templates with the model.
    """

    load_path: Optional[str]
    save_path: Optional[str]
    encode_batch_size: int

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> TemplateEmbeddingConfig:
        """Parse from ``cfg["training"]["stage2"]["template_embeddings"]``.

        Args:
            raw: The ``stage2.template_embeddings`` sub-dict (may be empty).

        Returns:
            Fully populated :class:`TemplateEmbeddingConfig` instance.
        """
        return cls(
            load_path=raw.get("load_path"),
            save_path=raw.get("save_path"),
            encode_batch_size=int(raw.get("encode_batch_size", 1024)),
        )


@dataclass
class TemplateCacheConfig:
    """Parsed template token feature cache settings.

    Attributes:
        path: File path for the tokenised feature cache.
        load_if_exists: Whether to load the cache if the file exists.
        save: Whether to write the cache to disk after building.
    """

    path: Optional[str]
    load_if_exists: bool
    save: bool

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> TemplateCacheConfig:
        """Parse from ``cfg["training"]["stage2"]["template_token_cache"]``.

        Args:
            raw: The ``stage2.template_token_cache`` sub-dict (may be empty).

        Returns:
            Fully populated :class:`TemplateCacheConfig` instance.
        """
        return cls(
            path=raw.get("path"),
            load_if_exists=bool(raw.get("load_if_exists", True)),
            save=bool(raw.get("save", False)),
        )


@dataclass
class FreezeConfig:
    """Normalized encoder freeze schedule for Stage-2.

    Attributes:
        mode: ``"fixed"`` or ``"alternate"``.
        template_frozen: Whether the template encoder starts frozen (fixed mode).
        product_frozen: Whether the product encoder starts frozen.
        alternate: Whether to alternate template freeze/unfreeze each epoch window.
        template_frozen_epochs: Number of consecutive epochs with template frozen
            (alternate mode).
        template_unfrozen_epochs: Number of consecutive epochs with template
            unfrozen (alternate mode).
        start_with_template_frozen: Phase of the alternating schedule at epoch 0.
        start_epoch: Epoch at which the alternating schedule begins.
    """

    mode: str
    template_frozen: bool
    product_frozen: bool
    alternate: bool
    template_frozen_epochs: int
    template_unfrozen_epochs: int
    start_with_template_frozen: bool
    start_epoch: int

    @classmethod
    def fixed(cls, *, template_frozen: bool, product_frozen: bool) -> FreezeConfig:
        """Build a fixed (non-alternating) freeze config.

        Args:
            template_frozen: Whether the template encoder is frozen.
            product_frozen: Whether the product encoder is frozen.

        Returns:
            :class:`FreezeConfig` with ``mode="fixed"``.
        """
        return cls(
            mode="fixed",
            template_frozen=template_frozen,
            product_frozen=product_frozen,
            alternate=False,
            template_frozen_epochs=1,
            template_unfrozen_epochs=1,
            start_with_template_frozen=True,
            start_epoch=0,
        )

    @classmethod
    def from_dict(cls, raw: Any, *, default_product_frozen: bool) -> FreezeConfig:
        """Parse from ``cfg["training"]["stage2"]["freeze_template_encoder"]``.

        Supports three formats:
        - ``bool``: simple fixed freeze flag.
        - ``dict`` with ``enabled=False``: no freezing.
        - ``dict`` with ``alternate=True``: alternating schedule.

        Args:
            raw: The raw config value (bool or dict).
            default_product_frozen: Product-encoder freeze flag from the
                sibling ``freeze_product_encoder`` key.

        Returns:
            Fully populated :class:`FreezeConfig` instance.
        """
        if isinstance(raw, bool):
            return cls.fixed(template_frozen=raw, product_frozen=default_product_frozen)

        if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
            return cls.fixed(template_frozen=False, product_frozen=default_product_frozen)

        if bool(raw.get("alternate", False)):
            return cls(
                mode="alternate",
                template_frozen=False,
                product_frozen=False,
                alternate=True,
                template_frozen_epochs=max(1, int(raw.get("template_frozen_epochs", 1))),
                template_unfrozen_epochs=max(1, int(raw.get("template_unfrozen_epochs", 1))),
                start_with_template_frozen=bool(raw.get("start_with_template_frozen", True)),
                start_epoch=max(0, int(raw.get("start_epoch", 0))),
            )

        return cls.fixed(
            template_frozen=bool(raw.get("template_frozen", raw.get("enabled", False))),
            product_frozen=bool(raw.get("product_frozen", default_product_frozen)),
        )


@dataclass
class Stage2Config:
    """All Stage-2 configuration parsed from the experiment config dict.

    Parsed once at module construction; all components receive typed sub-configs
    instead of drilling into raw ``Dict[str, Any]`` objects.

    Attributes:
        loss: Loss function hyper-parameters.
        candidate_sampling: Candidate construction hyper-parameters.
        evaluation: Validation evaluation settings.
        template_embeddings: Template embedding cache settings.
        template_token_cache: Template token feature cache settings.
        freeze: Encoder freeze schedule.
        keep_embeddings_on_gpu: Whether to keep the full template embedding
            matrix resident on GPU during training.
    """

    loss: LossConfig
    candidate_sampling: CandidateSamplingConfig
    evaluation: EvaluationConfig
    template_embeddings: TemplateEmbeddingConfig
    template_token_cache: TemplateCacheConfig
    freeze: FreezeConfig
    keep_embeddings_on_gpu: bool

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any], fallback_temperature: float) -> Stage2Config:
        """Parse from the full experiment config dict.

        Args:
            cfg: Full experiment config (top-level dict).
            fallback_temperature: Temperature from ``cfg["training"]`` used when
                ``stage2.loss.temperature`` is not set.

        Returns:
            Fully populated :class:`Stage2Config` instance.
        """
        raw = cfg.get("training", {}).get("stage2", {})
        sampling_cfg = CandidateSamplingConfig.from_dict(raw.get("candidate_sampling", {}))
        return cls(
            loss=LossConfig.from_dict(raw.get("loss", {}), fallback_temperature),
            candidate_sampling=sampling_cfg,
            evaluation=EvaluationConfig.from_dict(
                raw.get("evaluation", {}), sampling_cfg.candidate_size
            ),
            template_embeddings=TemplateEmbeddingConfig.from_dict(
                raw.get("template_embeddings", {})
            ),
            template_token_cache=TemplateCacheConfig.from_dict(
                raw.get("template_token_cache", {})
            ),
            freeze=FreezeConfig.from_dict(
                raw.get("freeze_template_encoder", True),
                default_product_frozen=bool(raw.get("freeze_product_encoder", False)),
            ),
            keep_embeddings_on_gpu=bool(raw.get("keep_template_embeddings_on_gpu", False)),
        )
