"""Data processing and Lightning data modules for reaction retrosynthesis.

This package merges what was previously split between ``src/data/`` and
``src/lightning_data_modules/``.  All public symbols are re-exported here
for convenient single-import access.
"""

from .datatypes import (
    CollatedFeatureDict,
    FeatureCollator,
    FeatureDict,
    FilterSummary,
    InputBuilder,
    JsonRow,
    RowValidator,
    ValidatorFactory,
)
from .datasets import (
    ReactionDataset,
    Stage2ProductDataset,
    Stage2TemplateLibrary,
    collect_template_library,
    iter_jsonl_rows,
    load_jsonl,
    reaction_collate_fn,
    stage2_collate_fn,
)
from .input_processing import (
    collate_feature_dicts,
    get_feature_collator,
    get_text_input_builder,
    tensorize_feature_dict,
)
from .tokenizer import CharTokenizer
from .validation import (
    build_row_validator_from_config,
    validate_dataset_jsonl,
    validate_product_template_row,
)
from .base import BaseReactionDataModule
from .factory import create_data_module
from .stage1 import Stage1ReactionDataModule
from .stage2 import Stage2ReactionDataModule

__all__ = [
    # datatypes
    "CollatedFeatureDict",
    "FeatureCollator",
    "FeatureDict",
    "FilterSummary",
    "InputBuilder",
    "JsonRow",
    "RowValidator",
    "ValidatorFactory",
    # datasets
    "ReactionDataset",
    "Stage2ProductDataset",
    "Stage2TemplateLibrary",
    "collect_template_library",
    "iter_jsonl_rows",
    "load_jsonl",
    "reaction_collate_fn",
    "stage2_collate_fn",
    # input processing
    "collate_feature_dicts",
    "get_feature_collator",
    "get_text_input_builder",
    "tensorize_feature_dict",
    # tokenizer
    "CharTokenizer",
    # validation
    "build_row_validator_from_config",
    "validate_dataset_jsonl",
    "validate_product_template_row",
    # data modules
    "BaseReactionDataModule",
    "create_data_module",
    "Stage1ReactionDataModule",
    "Stage2ReactionDataModule",
]
