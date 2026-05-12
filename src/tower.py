from __future__ import annotations

from typing import Any, Dict


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalize_pretrained_cfg(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    path = raw.get("path") or raw.get("checkpoint")
    if not path:
        return {}
    out: Dict[str, Any] = {"path": path}
    if "strict" in raw:
        out["strict"] = bool(raw["strict"])
    if "state_dict_key" in raw and raw["state_dict_key"] is not None:
        out["state_dict_key"] = raw["state_dict_key"]
    if "key_prefix" in raw and raw["key_prefix"] is not None:
        out["key_prefix"] = raw["key_prefix"]
    if "key_prefixes" in raw and isinstance(raw["key_prefixes"], list):
        out["key_prefixes"] = [str(x) for x in raw["key_prefixes"] if x is not None]
    return out


def _legacy_pretrained_cfg(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "pretrained" in raw and isinstance(raw["pretrained"], dict):
        return _normalize_pretrained_cfg(raw["pretrained"])
    path = raw.get("pretrained_path")
    if not path:
        return {}
    out: Dict[str, Any] = {"path": path}
    if "pretrained_strict" in raw:
        out["strict"] = bool(raw["pretrained_strict"])
    if raw.get("pretrained_state_dict_key") is not None:
        out["state_dict_key"] = raw["pretrained_state_dict_key"]
    if raw.get("pretrained_key_prefix") is not None:
        out["key_prefix"] = raw["pretrained_key_prefix"]
    return out


def resolve_tower_cfg(cfg: Dict[str, Any], kind: str) -> Dict[str, Any]:
    if kind not in ("product", "template"):
        raise ValueError(f"Unknown tower kind: {kind}")

    model_cfg = _as_dict(cfg.get("model", {}))
    legacy = _as_dict(model_cfg.get(f"{kind}_encoder", {}))
    tower = _as_dict(model_cfg.get(f"{kind}_tower", {}))

    encoder_cfg = _as_dict(tower.get("encoder", {}))
    preprocessing_cfg = _as_dict(tower.get("preprocessing", {}))
    projection_cfg = _as_dict(tower.get("projection", {}))

    # Allow compact style directly under model.{kind}_tower.
    if not encoder_cfg:
        compact_encoder = {
            "class_path": tower.get("class_path"),
            "kwargs": tower.get("kwargs"),
            "pretrained": tower.get("pretrained"),
        }
        encoder_cfg = {k: v for k, v in compact_encoder.items() if v is not None}
    if not preprocessing_cfg:
        compact_pre = {
            "input_builder": tower.get("input_builder"),
            "input_collator": tower.get("input_collator"),
        }
        preprocessing_cfg = {k: v for k, v in compact_pre.items() if v is not None}
    if not projection_cfg and isinstance(tower.get("projection"), dict):
        projection_cfg = dict(tower["projection"])

    legacy_encoder: Dict[str, Any] = {}
    if legacy.get("class_path") is not None:
        legacy_encoder["class_path"] = legacy["class_path"]
    if isinstance(legacy.get("kwargs"), dict):
        legacy_encoder["kwargs"] = dict(legacy["kwargs"])
    legacy_pretrained = _legacy_pretrained_cfg(legacy)
    if legacy_pretrained:
        legacy_encoder["pretrained"] = legacy_pretrained

    legacy_preprocessing: Dict[str, Any] = {}
    if legacy.get("input_builder") is not None:
        legacy_preprocessing["input_builder"] = legacy["input_builder"]
    if legacy.get("input_collator") is not None:
        legacy_preprocessing["input_collator"] = legacy["input_collator"]

    legacy_projection = {}
    if isinstance(legacy.get("projection"), dict):
        legacy_projection = dict(legacy["projection"])
    for key, legacy_key in (
        ("enabled", "projection_enabled"),
        ("expected_dim", "projection_expected_dim"),
        ("expand_dim", "projection_expand_dim"),
        ("dropout", "projection_dropout"),
    ):
        if legacy_key in legacy and key not in legacy_projection:
            legacy_projection[key] = legacy[legacy_key]

    merged_encoder = {**legacy_encoder, **encoder_cfg}
    if "pretrained" in merged_encoder:
        merged_encoder["pretrained"] = _normalize_pretrained_cfg(_as_dict(merged_encoder["pretrained"]))
    if not isinstance(merged_encoder.get("kwargs"), dict):
        merged_encoder["kwargs"] = {}

    merged_preprocessing = {**legacy_preprocessing, **preprocessing_cfg}
    merged_projection = {**legacy_projection, **projection_cfg}

    return {
        "kind": kind,
        "encoder": merged_encoder,
        "preprocessing": merged_preprocessing,
        "projection": merged_projection,
        "name": str(tower.get("name") or legacy.get("name") or merged_encoder.get("class_path") or "bert"),
    }
