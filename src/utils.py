import os
import json
import random
from typing import Dict, Any, Type
import importlib

import numpy as np
import torch
import yaml

from .data.tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_object(path: str):
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _tokenizer_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tok_cfg = cfg.get("tokenizer", {})
    return tok_cfg if isinstance(tok_cfg, dict) else {}


def tokenizer_enabled(cfg: Dict[str, Any]) -> bool:
    tok_cfg = _tokenizer_cfg(cfg)
    return bool(tok_cfg) and bool(tok_cfg.get("enabled", True))


def resolve_tokenizer_class(cfg: Dict[str, Any]) -> Type[CharTokenizer]:
    tok_cfg = _tokenizer_cfg(cfg)
    class_path = tok_cfg.get("class_path")
    if class_path:
        return load_object(class_path)
    return CharTokenizer


def load_or_build_tokenizer(cfg: Dict[str, Any], allow_build: bool) -> Any:
    if not tokenizer_enabled(cfg):
        return None

    tok_cfg = _tokenizer_cfg(cfg)
    tokenizer_cls = resolve_tokenizer_class(cfg)
    vocab_path = tok_cfg.get("vocab_path")
    if not vocab_path:
        raise RuntimeError("tokenizer.vocab_path is required when tokenizer is enabled.")

    if os.path.exists(vocab_path) and not tok_cfg.get("rebuild", False):
        return tokenizer_cls.load(vocab_path)

    if not allow_build:
        raise RuntimeError(f"Tokenizer vocab file does not exist: {vocab_path}")

    fields = tok_cfg.get("fields", ["product", "template"])
    paths = tok_cfg.get("build_from_list")
    if paths:
        tokenizer = tokenizer_cls.build_from_jsonl_files(paths, fields)
    else:
        if "build_from" not in tok_cfg:
            raise RuntimeError(
                "Provide tokenizer.build_from or tokenizer.build_from_list when tokenizer build is needed."
            )
        tokenizer = tokenizer_cls.build_from_jsonl_files([tok_cfg["build_from"]], fields=fields)

    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    tokenizer.save(vocab_path)
    return tokenizer


def load_checkpoint_payload(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint payload at '{path}' is not a dict.")
    return payload


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not prefix:
        return dict(state_dict)
    return {k[len(prefix) :] if isinstance(k, str) and k.startswith(prefix) else k: v for k, v in state_dict.items()}


def load_weights_into_lightning_module(module: Any, checkpoint_path: str) -> None:
    ckpt = load_checkpoint_payload(checkpoint_path)
    if "state_dict" in ckpt:
        module.load_state_dict(ckpt["state_dict"], strict=False)
    elif "model_state" in ckpt:
        module.model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        module.load_state_dict(ckpt, strict=False)


def load_weights_into_model(model: Any, checkpoint_path: str, lightning_prefix: str = "model.") -> None:
    ckpt = load_checkpoint_payload(checkpoint_path)
    if "state_dict" in ckpt:
        state = _strip_prefix_if_present(ckpt["state_dict"], lightning_prefix)
        model.load_state_dict(state, strict=False)
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
