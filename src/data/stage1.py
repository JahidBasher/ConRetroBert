"""Stage 1 PyTorch Lightning data module.

Loads paired (product SMILES, template SMARTS) data for contrastive or
classification training in Stage 1.
"""

from functools import partial
from typing import Any, Dict, Optional

from .base import BaseReactionDataModule
from .datasets import ReactionDataset, reaction_collate_fn
from .datatypes import ValidatorFactory


class Stage1ReactionDataModule(BaseReactionDataModule):
    """Data module for Stage 1 reaction template training.

    Builds :class:`~data.datasets.ReactionDataset` instances for the train
    and optional validation splits, wires up injected product and template
    input builders and collators, and delegates DataLoader construction to
    :class:`~data.base.BaseReactionDataModule`.

    Inherits ``__init__`` from :class:`~data.base.BaseReactionDataModule`
    directly — no additional initialisation is required.

    Args:
        cfg: Full experiment config dict.
        tokenizer: CharTokenizer or compatible encoder; may be None when
            custom input builders are configured.
        validator_factory: Optional injected :data:`ValidatorFactory`.
            Defaults to reading ``cfg["data"]["validation_filter"]``.
    """

    def setup(self, stage: Optional[str] = None) -> None:
        """Build train and optional validation datasets.

        Skips setup when *stage* is not None or "fit" (e.g. "test", "predict").

        Args:
            stage: Lightning stage string passed by the Trainer.
        """
        if stage not in (None, "fit"):
            return

        dcfg = self.cfg["data"]
        tcfg = self.cfg.get("tokenizer", {})
        product_input_builder, product_collator = self._build_product_inputs()
        template_input_builder, template_collator = self._build_template_inputs()
        self.collate_fn = partial(
            reaction_collate_fn,
            product_collator=product_collator,
            template_collator=template_collator,
        )

        self.train_dataset = ReactionDataset(
            path=dcfg["train_path"],
            tokenizer=self.tokenizer,
            max_product_len=int(tcfg.get("max_product_len", 0)),
            max_template_len=int(tcfg.get("max_template_len", 0)),
            add_bos_eos=tcfg.get("add_bos_eos", False),
            limit=dcfg.get("limit"),
            product_input_builder=product_input_builder,
            template_input_builder=template_input_builder,
            row_validator=self.validator_factory("train"),
        )
        self._log_row_filter_summary("train", self.train_dataset)

        if dcfg.get("val_path"):
            self.val_dataset = ReactionDataset(
                path=dcfg["val_path"],
                tokenizer=self.tokenizer,
                max_product_len=int(tcfg.get("max_product_len", 0)),
                max_template_len=int(tcfg.get("max_template_len", 0)),
                add_bos_eos=tcfg.get("add_bos_eos", False),
                limit=dcfg.get("val_limit"),
                product_input_builder=product_input_builder,
                template_input_builder=template_input_builder,
                row_validator=self.validator_factory("val"),
            )
            self._log_row_filter_summary("val", self.val_dataset)
