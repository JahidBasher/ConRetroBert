"""Stage 2 PyTorch Lightning data module.

Loads product-centric data for template ranking training in Stage 2.
Builds a :class:`~data.datasets.Stage2TemplateLibrary` from the configured
library path, then constructs :class:`~data.datasets.Stage2ProductDataset`
instances for the train and optional validation splits.
"""

from functools import partial
from typing import Any, Dict, Optional

from .base import BaseReactionDataModule
from .datasets import (
    Stage2ProductDataset,
    Stage2TemplateLibrary,
    collect_template_library,
    stage2_collate_fn,
)
from .datatypes import ValidatorFactory


class Stage2ReactionDataModule(BaseReactionDataModule):
    """Data module for Stage 2 template ranking training.

    Args:
        cfg: Full experiment config dict.
        tokenizer: CharTokenizer or compatible encoder; may be None when a
            custom product input builder is configured.
        validator_factory: Optional injected :data:`ValidatorFactory`.
            Defaults to reading ``cfg["data"]["validation_filter"]``.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        tokenizer: Any,
        validator_factory: Optional[ValidatorFactory] = None,
    ) -> None:
        super().__init__(cfg, tokenizer, validator_factory)
        self.template_library: Optional[Stage2TemplateLibrary] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Build the template library and train/validation datasets.

        Reads ``cfg["training"]["stage2"]`` for library path, applicable
        template settings, and per-split aggregation flags.
        Skips setup when *stage* is not None or "fit".

        Args:
            stage: Lightning stage string passed by the Trainer.
        """
        if stage not in (None, "fit"):
            return

        dcfg = self.cfg["data"]
        tcfg = self.cfg.get("tokenizer", {})
        s2cfg = self.cfg["training"].get("stage2", {})

        product_input_builder, product_collator = self._build_product_inputs()
        self.collate_fn = partial(stage2_collate_fn, product_collator=product_collator)

        template_library_path = s2cfg.get("template_library_path") or dcfg["train_path"]
        self.template_library = Stage2TemplateLibrary(
            collect_template_library(template_library_path, limit=s2cfg.get("template_library_limit"))
        )

        appl_cfg = s2cfg.get("applicable_templates", {})
        appl_path = appl_cfg.get("path") if appl_cfg.get("enabled", False) else None
        max_appl = int(appl_cfg.get("max_per_product", 512))

        self.train_dataset = Stage2ProductDataset(
            path=dcfg["train_path"],
            tokenizer=self.tokenizer,
            template_to_id=self.template_library.template_to_id,
            max_product_len=int(tcfg.get("max_product_len", 0)),
            add_bos_eos=tcfg.get("add_bos_eos", False),
            limit=dcfg.get("limit"),
            aggregate_by_product=s2cfg.get("aggregate_train_by_product", True),
            product_input_builder=product_input_builder,
            row_validator=self.validator_factory("train"),
            appl_path=appl_path,
            max_appl_per_product=max_appl,
        )
        self._log_row_filter_summary("train", self.train_dataset)

        if dcfg.get("val_path"):
            self.val_dataset = Stage2ProductDataset(
                path=dcfg["val_path"],
                tokenizer=self.tokenizer,
                template_to_id=self.template_library.template_to_id,
                max_product_len=int(tcfg.get("max_product_len", 0)),
                add_bos_eos=tcfg.get("add_bos_eos", False),
                limit=dcfg.get("val_limit"),
                aggregate_by_product=s2cfg.get("aggregate_val_by_product", True),
                product_input_builder=product_input_builder,
                row_validator=self.validator_factory("val"),
                appl_path=appl_path,
                max_appl_per_product=max_appl,
            )
            self._log_row_filter_summary("val", self.val_dataset)
