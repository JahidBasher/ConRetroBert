"""Type definitions for the data/ subdirectory.

All callable protocols, type aliases, and structured dict types used across
datasets, input processing, validation, and data modules are centralised here.
Import from this module instead of re-declaring types in individual files.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Primitive row / feature types
# ---------------------------------------------------------------------------

# A single parsed row from a JSONL data file.
JsonRow = Dict[str, Any]

# Output of an encoder: typically {"input_ids": [...], "attention_mask": [...]}.
FeatureDict = Dict[str, Any]

# A FeatureDict whose values have all been stacked into tensors.
CollatedFeatureDict = Dict[str, torch.Tensor]

# Summary dict emitted by dataset classes describing how many rows passed filtering.
FilterSummary = Dict[str, Any]


# ---------------------------------------------------------------------------
# Callable type aliases
# ---------------------------------------------------------------------------

# Encodes a SMILES or SMARTS string into a FeatureDict.
InputBuilder = Callable[[str], FeatureDict]

# Collates a list of FeatureDicts into a single CollatedFeatureDict.
FeatureCollator = Callable[[List[FeatureDict]], CollatedFeatureDict]

# Validates a single data row; returns (keep: bool, reason: str).
RowValidator = Callable[[JsonRow], Tuple[bool, str]]

# Accepts a split name ("train", "val", "test") and returns an optional RowValidator.
# Returning None means all rows are accepted for that split.
ValidatorFactory = Callable[[str], Optional[RowValidator]]
