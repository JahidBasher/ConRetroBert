"""Input builder and feature collator factories for product and template towers.

Resolves the preprocessing callables for each encoder tower from the
experiment config, supporting both dynamically-loaded custom callables and
the default CharTokenizer-based path.

Key public functions:

- :func:`get_text_input_builder`: returns an :data:`InputBuilder` for a tower.
- :func:`get_feature_collator`: returns a :data:`FeatureCollator` for a tower.
- :func:`collate_feature_dicts`: default collation with variable-length padding.
- :func:`tensorize_feature_dict`: convert a single feature dict to tensors.
"""

import importlib
from typing import Any, Dict, List

import torch
from torch.utils.data._utils.collate import default_collate

from ..tower import resolve_tower_cfg
from .datatypes import FeatureCollator, FeatureDict, InputBuilder


def load_object(path: str) -> Any:
    """Dynamically import and return an object from a dotted or colon-separated path.

    Supports both ``module.submodule.attr`` and ``module.submodule:attr``
    notation for flexibility with entry-point style paths.

    Args:
        path: Import path string identifying the object.

    Returns:
        The resolved Python object (function, class, etc.).
    """
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def get_text_input_builder(cfg: Dict[str, Any], tokenizer: Any, kind: str) -> InputBuilder:
    """Build the input-encoding callable for a product or template encoder tower.

    If the tower config specifies ``preprocessing.input_builder``, that
    callable is loaded dynamically and wrapped to enforce the return type.
    Otherwise a closure over the provided *tokenizer* is returned, configured
    from ``cfg["tokenizer"]``.

    Args:
        cfg: Full experiment config dict.
        tokenizer: CharTokenizer instance (required on the default path).
        kind: "product" or "template".

    Returns:
        An :data:`InputBuilder` callable ``(text: str) -> FeatureDict``.

    Raises:
        ValueError: If *kind* is not "product" or "template".
        RuntimeError: If no tokenizer and no custom builder are configured.
    """
    if kind not in ("product", "template"):
        raise ValueError(f"Unknown kind: {kind!r}")

    tower_cfg = resolve_tower_cfg(cfg, kind)
    builder_path = tower_cfg.get("preprocessing", {}).get("input_builder")

    if builder_path:
        fn = load_object(builder_path)

        def _custom_builder(text: str) -> FeatureDict:
            out = fn(text=text, kind=kind, tokenizer=tokenizer, cfg=cfg)
            if not isinstance(out, dict):
                raise RuntimeError(
                    f"Input builder {builder_path!r} must return a dict, got {type(out)}"
                )
            return out

        return _custom_builder

    if tokenizer is None:
        raise RuntimeError(
            f"No tokenizer is available for {kind} tower and no custom input_builder is configured. "
            f"Set model.{kind}_tower.preprocessing.input_builder (or legacy "
            f"model.{kind}_encoder.input_builder) or enable tokenizer."
        )

    tcfg = cfg.get("tokenizer", {})
    max_len = int(tcfg["max_product_len"] if kind == "product" else tcfg["max_template_len"])
    add_bos_eos = bool(tcfg.get("add_bos_eos", False))
    pad_to_max_length = bool(tcfg.get("pad_to_max_length", False))

    def _default_builder(text: str) -> FeatureDict:
        return tokenizer.encode(
            text,
            kind=kind,
            max_length=max_len,
            add_bos_eos=add_bos_eos,
            pad_to_max_length=pad_to_max_length,
        )

    return _default_builder


def tensorize_feature_dict(features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Convert all values in a feature dict to tensors.

    Values that are already tensors are kept as-is; all others are converted
    with ``torch.as_tensor``.

    Args:
        features: Dict whose values are lists, scalars, or existing tensors.

    Returns:
        New dict with identical keys and all-tensor values.
    """
    out: Dict[str, torch.Tensor] = {}
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            out[key] = value
        else:
            out[key] = torch.as_tensor(value)
    return out


def _pad_value_for_key(key: str) -> int:
    """Return the appropriate padding fill value for a given feature key.

    Labels use -100 (the PyTorch cross-entropy ignore index); all other
    keys use 0 (standard attention/padding mask fill value).

    Args:
        key: Feature key name.

    Returns:
        Integer fill value: -100 for label keys, 0 for all others.
    """
    normalized = str(key).lower()
    if normalized == "labels" or normalized.endswith("_labels"):
        return -100
    return 0


def _collate_tensor_values(values: List[torch.Tensor], key: str) -> torch.Tensor:
    """Stack or pad a list of tensors along a new batch dimension.

    Handles four cases:
    - All scalars: stacked directly.
    - All equal shape: stacked without padding.
    - Variable-length 1-D sequences with equal tail dims: right-padded.
    - Mismatched trailing dimensions: delegated to PyTorch ``default_collate``.

    Args:
        values: Non-empty list of tensors with compatible dtypes.
        key: Feature key name (determines the pad fill value via
            :func:`_pad_value_for_key`).

    Returns:
        Batched tensor of shape (len(values), ...).

    Raises:
        RuntimeError: If *values* is empty.
    """
    if not values:
        raise RuntimeError("Cannot collate empty tensor list.")

    if all(v.dim() == 0 for v in values):
        return torch.stack(values, dim=0)

    shapes = [tuple(v.shape) for v in values]
    if len(set(shapes)) == 1:
        return torch.stack(values, dim=0)

    tail_shape = values[0].shape[1:]
    if not all(v.shape[1:] == tail_shape for v in values):
        return default_collate(values)

    max_len = max(int(v.shape[0]) for v in values)
    out_shape = (len(values), max_len) + tail_shape
    out = torch.full(out_shape, fill_value=_pad_value_for_key(key), dtype=values[0].dtype)
    for i, v in enumerate(values):
        n = int(v.shape[0])
        if n == 0:
            continue
        out[i, :n] = v
    return out


def collate_feature_dicts(features: List[FeatureDict]) -> Dict[str, torch.Tensor]:
    """Collate a list of feature dicts into a single batched dict of tensors.

    All keys present in the first dict are expected in every subsequent dict.
    Values are converted to tensors then batched with right-padding where
    sequence lengths differ.

    Args:
        features: Non-empty list of feature dicts with consistent keys.

    Returns:
        Dict mapping each key to a batched tensor.

    Raises:
        RuntimeError: If *features* is empty.
    """
    if not features:
        raise RuntimeError("Cannot collate empty feature list.")

    out: Dict[str, torch.Tensor] = {}
    keys = features[0].keys()
    for key in keys:
        values = [x[key] for x in features]
        tensor_values: List[torch.Tensor] = [
            v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in values
        ]
        out[key] = _collate_tensor_values(tensor_values, key=key)
    return out


def get_feature_collator(cfg: Dict[str, Any], kind: str) -> FeatureCollator:
    """Build the feature-collation callable for a product or template encoder tower.

    If the tower config specifies ``preprocessing.input_collator``, that
    callable is loaded dynamically and wrapped to enforce the return type.
    Otherwise :func:`collate_feature_dicts` is returned directly.

    Args:
        cfg: Full experiment config dict.
        kind: "product" or "template".

    Returns:
        A :data:`FeatureCollator` callable
        ``(features: List[FeatureDict]) -> CollatedFeatureDict``.

    Raises:
        ValueError: If *kind* is not "product" or "template".
    """
    if kind not in ("product", "template"):
        raise ValueError(f"Unknown kind: {kind!r}")

    tower_cfg = resolve_tower_cfg(cfg, kind)
    collator_path = tower_cfg.get("preprocessing", {}).get("input_collator")

    if collator_path:
        fn = load_object(collator_path)

        def _custom_collator(features: List[FeatureDict]) -> Dict[str, Any]:
            out = fn(features=features, kind=kind, cfg=cfg)
            if not isinstance(out, dict):
                raise RuntimeError(
                    f"Input collator {collator_path!r} must return a dict, got {type(out)}"
                )
            return out

        return _custom_collator

    return collate_feature_dicts
