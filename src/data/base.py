"""Base PyTorch Lightning data module for reaction retrosynthesis training.

Provides DataLoader construction, per-split validator building, filter-summary
logging, and helpers to assemble per-tower input encoders and collators.
Stage-specific subclasses only need to implement ``setup()``.
"""

import functools
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datatypes import FeatureCollator, InputBuilder, ValidatorFactory
from .input_processing import get_feature_collator, get_text_input_builder
from .validation import build_row_validator_from_config


class BaseReactionDataModule(pl.LightningDataModule):
    """Shared base for Stage 1 and Stage 2 data modules.

    Handles DataLoader construction (batch size, num_workers, pin_memory, etc.)
    and provides helpers to build per-tower input encoders/collators from the
    experiment config.

    **Dependency Inversion (DIP):** the *validator_factory* parameter lets
    callers inject any callable that maps a split name to an optional
    :data:`RowValidator`.  This decouples the data module from the concrete
    config-parsing logic in :func:`build_row_validator_from_config` and makes
    the class easy to test with mock validators.  When omitted, the
    config-driven default is used via ``functools.partial``.

    Args:
        cfg: Full experiment config dict.
        tokenizer: CharTokenizer or compatible duck-typed encoder; may be None
            when custom input builders are configured for all towers.
        validator_factory: Callable ``(split: str) -> Optional[RowValidator]``.
            Defaults to reading ``cfg["data"]["validation_filter"]`` via
            :func:`build_row_validator_from_config`.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        tokenizer: Any,
        validator_factory: Optional[ValidatorFactory] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.validator_factory: ValidatorFactory = validator_factory or functools.partial(
            build_row_validator_from_config, cfg
        )
        self.train_dataset = None
        self.val_dataset = None
        self.collate_fn = None

    def _build_product_inputs(self) -> Tuple[InputBuilder, FeatureCollator]:
        """Build the product input encoder and feature collator from config.

        Returns:
            Tuple of (product_input_builder, product_collator).
        """
        builder = get_text_input_builder(self.cfg, self.tokenizer, "product")
        collator = get_feature_collator(self.cfg, "product")
        return builder, collator

    def _build_template_inputs(self) -> Tuple[InputBuilder, FeatureCollator]:
        """Build the template input encoder and feature collator from config.

        Returns:
            Tuple of (template_input_builder, template_collator).
        """
        builder = get_text_input_builder(self.cfg, self.tokenizer, "template")
        collator = get_feature_collator(self.cfg, "template")
        return builder, collator

    def _log_row_filter_summary(self, split: str, dataset: Any) -> None:
        """Print a one-line filter summary when row validation is enabled.

        Reads the ``filter_summary`` attribute from *dataset* and logs seen,
        kept, and dropped counts along with the top-5 drop reasons.

        Args:
            split: Split name used in the log prefix (e.g. "train", "val").
            dataset: Dataset object that exposes a ``filter_summary`` dict.
        """
        summary = getattr(dataset, "filter_summary", None)
        if not isinstance(summary, dict) or not summary.get("enabled", False):
            return

        seen_rows = int(summary.get("seen_rows", 0))
        kept_rows = int(summary.get("kept_rows", 0))
        dropped_rows = int(summary.get("dropped_rows", 0))
        reasons = summary.get("dropped_reason_counts", {})
        top: list = []
        if isinstance(reasons, dict):
            top = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_text = ", ".join([f"{k}:{v}" for k, v in top]) if top else "none"
        print(
            f"[data.validation_filter:{split}] seen={seen_rows} kept={kept_rows} "
            f"dropped={dropped_rows} top_reasons={top_text}"
        )

    def _build_loader(self, dataset: Any, shuffle: bool) -> DataLoader:
        """Construct a DataLoader with settings drawn from the experiment config.

        Reads ``cfg["data"]`` for worker count and loader knobs, and
        ``cfg["training"]["batch_size"]`` for the batch size.

        Args:
            dataset: A PyTorch Dataset instance.
            shuffle: Whether to shuffle the dataset each epoch.

        Returns:
            Configured DataLoader.
        """
        dcfg = self.cfg["data"]
        tcfg = self.cfg["training"]
        loader_cfg = dcfg.get("loader", {})
        num_workers = int(dcfg.get("num_workers", 0))
        kwargs: Dict[str, Any] = {}
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 2))

        return DataLoader(
            dataset,
            batch_size=tcfg["batch_size"],
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=bool(loader_cfg.get("pin_memory", True)),
            persistent_workers=bool(loader_cfg.get("persistent_workers", num_workers > 0)),
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """Return a shuffled DataLoader over the training dataset."""
        return self._build_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return an unshuffled DataLoader over the validation dataset, or None."""
        if self.val_dataset is None:
            return None
        return self._build_loader(self.val_dataset, shuffle=False)
