from __future__ import annotations

import warnings
from inspect import Parameter, signature
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .tower import resolve_tower_cfg
from .utils import load_object


class BertEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_length, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._gradient_checkpointing = False
        ff_dim = hidden_size * 4
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable/disable gradient checkpointing for transformer layers."""
        self._gradient_checkpointing = bool(enabled)

    @staticmethod
    def _layer_has_trainable_params(layer: nn.Module) -> bool:
        """Return True if a transformer block has any trainable parameter."""
        return any(p.requires_grad for p in layer.parameters())

    def _forward_transformer(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        if not self._gradient_checkpointing or not self.training:
            return self.encoder(x, src_key_padding_mask=key_padding_mask)

        out = x
        for layer in self.encoder.layers:
            if self._layer_has_trainable_params(layer):
                out = checkpoint(
                    lambda inp, _layer=layer: _layer(
                        inp,
                        src_mask=None,
                        src_key_padding_mask=key_padding_mask,
                        is_causal=False,
                    ),
                    out,
                    use_reentrant=False,
                )
            else:
                out = layer(
                    out,
                    src_mask=None,
                    src_key_padding_mask=key_padding_mask,
                    is_causal=False,
                )
        if self.encoder.norm is not None:
            out = self.encoder.norm(out)
        return out

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.norm(x)
        x = self.dropout(x)
        key_padding_mask = attention_mask == 0
        out = self._forward_transformer(x, key_padding_mask=key_padding_mask)
        return out


class ProjectionMLP(nn.Module):
    def __init__(self, expected_dim: int, expand_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.expand = nn.LazyLinear(expand_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.contract = nn.Linear(expand_dim, expected_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.act(x)
        x = self.drop(x)
        return self.contract(x)


def _coerce_tower_cfg(tower_cfg: Optional[Dict[str, Any]], encoder_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if tower_cfg and isinstance(tower_cfg, dict):
        out = dict(tower_cfg)
        if "encoder" not in out:
            out = {"encoder": dict(out)}
        if not isinstance(out.get("encoder"), dict):
            out["encoder"] = {}
        if not isinstance(out.get("projection"), dict):
            out["projection"] = {}
        return out
    return {
        "encoder": dict(encoder_cfg or {}),
        "projection": {},
        "preprocessing": {},
    }


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_product_len: int,
        max_template_len: int,
        shared_encoder: bool,
        mlm_enabled: bool,
        product_encoder_cfg: Optional[Dict[str, Any]] = None,
        template_encoder_cfg: Optional[Dict[str, Any]] = None,
        product_tower_cfg: Optional[Dict[str, Any]] = None,
        template_tower_cfg: Optional[Dict[str, Any]] = None,
        expected_embedding_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.shared_encoder = shared_encoder
        self.expected_embedding_dim = expected_embedding_dim

        self._init_tower_configs(
            product_tower_cfg=product_tower_cfg,
            template_tower_cfg=template_tower_cfg,
            product_encoder_cfg=product_encoder_cfg,
            template_encoder_cfg=template_encoder_cfg,
        )
        self._init_encoders(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_product_len=max_product_len,
            max_template_len=max_template_len,
        )
        self._init_heads(
            expected_embedding_dim=expected_embedding_dim,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            mlm_enabled=mlm_enabled,
        )

    def _init_tower_configs(
        self,
        product_tower_cfg: Optional[Dict[str, Any]],
        template_tower_cfg: Optional[Dict[str, Any]],
        product_encoder_cfg: Optional[Dict[str, Any]],
        template_encoder_cfg: Optional[Dict[str, Any]],
    ) -> None:
        self.product_tower_cfg = _coerce_tower_cfg(product_tower_cfg, product_encoder_cfg)
        self.template_tower_cfg = _coerce_tower_cfg(template_tower_cfg, template_encoder_cfg)
        self.product_encoder_cfg = dict(self.product_tower_cfg.get("encoder", {}))
        self.template_encoder_cfg = dict(self.template_tower_cfg.get("encoder", {}))

    def _init_encoders(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_product_len: int,
        max_template_len: int,
    ) -> None:
        if self.shared_encoder:
            self._init_shared_encoder(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_product_len=max_product_len,
                max_template_len=max_template_len,
            )
            return

        self._init_independent_encoders(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_product_len=max_product_len,
            max_template_len=max_template_len,
        )

    def _init_shared_encoder(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_product_len: int,
        max_template_len: int,
    ) -> None:
        if (
            self.product_encoder_cfg
            and self.template_encoder_cfg
            and self.product_encoder_cfg != self.template_encoder_cfg
        ):
            raise RuntimeError("When model.shared_encoder=true, product/template encoder configs must match.")

        merged_cfg = self.product_encoder_cfg or self.template_encoder_cfg
        encoder = self._build_encoder(
            encoder_cfg=merged_cfg,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_length=max(max_product_len, max_template_len),
        )
        self._maybe_load_pretrained_encoder(encoder, merged_cfg.get("pretrained", {}), kind="shared")
        self.product_encoder = encoder
        self.template_encoder = encoder

    def _init_independent_encoders(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_product_len: int,
        max_template_len: int,
    ) -> None:
        self.product_encoder = self._build_encoder(
            encoder_cfg=self.product_encoder_cfg,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_length=max_product_len,
        )
        self.template_encoder = self._build_encoder(
            encoder_cfg=self.template_encoder_cfg,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_length=max_template_len,
        )
        self._maybe_load_pretrained_encoder(
            self.product_encoder,
            self.product_encoder_cfg.get("pretrained", {}),
            kind="product",
        )
        self._maybe_load_pretrained_encoder(
            self.template_encoder,
            self.template_encoder_cfg.get("pretrained", {}),
            kind="template",
        )

    def _init_heads(
        self,
        expected_embedding_dim: Optional[int],
        hidden_size: int,
        vocab_size: int,
        mlm_enabled: bool,
    ) -> None:
        self.product_projector = self._build_projector(
            self.product_tower_cfg.get("projection", {}), expected_embedding_dim
        )
        self.template_projector = self._build_projector(
            self.template_tower_cfg.get("projection", {}), expected_embedding_dim
        )

        self.mlm_enabled = mlm_enabled
        self.mlm_head = nn.Linear(hidden_size, vocab_size) if mlm_enabled else None

    def _build_projector(self, projection_cfg: Dict[str, Any], expected_dim: Optional[int]) -> nn.Module:
        cfg = dict(projection_cfg or {})
        enabled = cfg.get("enabled")
        proj_dim = cfg.get("expected_dim", expected_dim)
        if enabled is None:
            enabled = proj_dim is not None
        if not enabled:
            return nn.Identity()
        if proj_dim is None:
            raise RuntimeError("Projection is enabled but expected_dim is not set.")
        proj_dim = int(proj_dim)
        if proj_dim <= 0:
            raise RuntimeError("Projection expected_dim must be > 0.")
        expand_dim = cfg.get("expand_dim")
        if expand_dim is None:
            factor = float(cfg.get("expand_factor", 2.0))
            expand_dim = max(proj_dim, int(round(proj_dim * factor)))
        expand_dim = int(expand_dim)
        if expand_dim <= 0:
            raise RuntimeError("Projection expand_dim must be > 0.")
        proj_dropout = float(cfg.get("dropout", 0.0))
        return ProjectionMLP(expected_dim=proj_dim, expand_dim=expand_dim, dropout=proj_dropout)

    def _build_encoder(
        self,
        encoder_cfg: Optional[Dict[str, Any]],
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_length: int,
    ) -> nn.Module:
        cfg = encoder_cfg or {}
        class_path = cfg.get("class_path")
        user_kwargs = dict(cfg.get("kwargs", {})) if isinstance(cfg.get("kwargs", {}), dict) else {}

        if not class_path:
            return BertEncoder(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_length=max_length,
            )

        encoder_cls = load_object(class_path)
        default_kwargs: Dict[str, Any] = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "max_length": max_length,
        }
        init_kwargs = {**default_kwargs, **user_kwargs}

        try:
            sig = signature(encoder_cls.__init__)
            accepts_kwargs = any(p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not accepts_kwargs:
                init_kwargs = {k: v for k, v in init_kwargs.items() if k in sig.parameters}
        except Exception:
            pass

        try:
            return encoder_cls(**init_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build encoder '{class_path}' with kwargs keys {sorted(init_kwargs.keys())}"
            ) from exc

    def _resolve_checkpoint_state_dict(self, payload: Any, state_dict_key: Optional[str]) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise RuntimeError("Pretrained checkpoint payload must be a mapping/dict.")
        if state_dict_key:
            if state_dict_key not in payload:
                raise RuntimeError(f"state_dict_key='{state_dict_key}' was not found in pretrained checkpoint.")
            state = payload[state_dict_key]
            if not isinstance(state, Mapping):
                raise RuntimeError(f"Checkpoint key '{state_dict_key}' does not contain a state dict mapping.")
            return state
        for key in ("model_state", "state_dict"):
            maybe_state = payload.get(key)
            if isinstance(maybe_state, Mapping):
                return maybe_state
        return payload

    def _prefix_candidates(self, kind: str, user_prefixes: Optional[Any]) -> Tuple[str, ...]:
        prefixes = []
        if isinstance(user_prefixes, str) and user_prefixes:
            prefixes.append(user_prefixes)
        elif isinstance(user_prefixes, list):
            prefixes.extend([str(x) for x in user_prefixes if x])

        defaults = {
            "product": (
                "model.product_encoder.",
                "product_encoder.",
                "model.product_tower.encoder.",
                "product_tower.encoder.",
            ),
            "template": (
                "model.template_encoder.",
                "template_encoder.",
                "model.template_tower.encoder.",
                "template_tower.encoder.",
            ),
            "shared": (
                "model.product_encoder.",
                "product_encoder.",
                "model.template_encoder.",
                "template_encoder.",
                "model.encoder.",
                "encoder.",
            ),
        }
        prefixes.extend(defaults.get(kind, ()))
        return tuple(prefixes)

    def _select_best_state_dict(
        self, encoder: nn.Module, raw_state: Mapping[str, Any], prefixes: Tuple[str, ...]
    ) -> Dict[str, Any]:
        encoder_keys = set(encoder.state_dict().keys())
        candidates = [dict(raw_state)]
        for prefix in prefixes:
            stripped = {
                k[len(prefix) :]: v for k, v in raw_state.items() if isinstance(k, str) and k.startswith(prefix)
            }
            if stripped:
                candidates.append(stripped)

        best = candidates[0]
        best_score = -1
        for cand in candidates:
            score = len(encoder_keys.intersection(set(cand.keys())))
            if score > best_score:
                best = cand
                best_score = score
        if best_score <= 0:
            raise RuntimeError("No matching encoder keys were found in pretrained checkpoint state dict.")
        return best

    def _maybe_load_pretrained_encoder(self, encoder: nn.Module, pretrained_cfg: Any, kind: str) -> None:
        if not isinstance(pretrained_cfg, Mapping):
            return
        path = pretrained_cfg.get("path")
        if not path:
            return
        payload = torch.load(path, map_location="cpu")
        state = self._resolve_checkpoint_state_dict(payload, pretrained_cfg.get("state_dict_key"))
        state = {str(k): v for k, v in state.items()}
        prefixes = self._prefix_candidates(
            kind=kind,
            user_prefixes=pretrained_cfg.get("key_prefixes") or pretrained_cfg.get("key_prefix"),
        )
        best_state = self._select_best_state_dict(encoder, state, prefixes=prefixes)
        strict = bool(pretrained_cfg.get("strict", False))
        incompatible = encoder.load_state_dict(best_state, strict=strict)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            warnings.warn(
                f"Loaded pretrained weights for {kind} encoder with missing={len(incompatible.missing_keys)} "
                f"and unexpected={len(incompatible.unexpected_keys)} keys (strict={strict}).",
                stacklevel=2,
            )

    def _normalize_inputs(
        self,
        inputs_or_ids: Any,
        attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if isinstance(inputs_or_ids, Mapping):
            return dict(inputs_or_ids)
        if attention_mask is None:
            raise RuntimeError("attention_mask is required when passing tensor inputs instead of a dict.")
        return {
            "input_ids": inputs_or_ids,
            "attention_mask": attention_mask,
        }

    def _call_encoder_module(self, encoder: nn.Module, inputs: Dict[str, torch.Tensor]) -> Any:
        if isinstance(encoder, BertEncoder):
            if "input_ids" not in inputs or "attention_mask" not in inputs:
                raise RuntimeError("BertEncoder requires input_ids and attention_mask.")
            return encoder(inputs["input_ids"], inputs["attention_mask"])

        if hasattr(encoder, "encode") and callable(getattr(encoder, "encode")):
            encode_fn = getattr(encoder, "encode")
            return self._call_with_fallback(encode_fn, inputs)

        return self._call_with_fallback(encoder, inputs)

    def _call_with_fallback(self, fn: Any, inputs: Dict[str, torch.Tensor]) -> Any:
        accepts_kwargs = False
        supported: Dict[str, torch.Tensor] = {}
        try:
            sig = signature(fn if not isinstance(fn, nn.Module) else fn.forward)
            params = sig.parameters
            accepts_kwargs = any(p.kind == Parameter.VAR_KEYWORD for p in params.values())
            supported = {k: v for k, v in inputs.items() if k in params}
        except (TypeError, ValueError):
            pass

        if accepts_kwargs:
            try:
                return fn(**inputs)
            except TypeError:
                pass
        elif supported:
            try:
                return fn(**supported)
            except TypeError:
                pass
        return fn(inputs)

    def _run_tower_encoder(
        self, encoder: nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        out = self._call_encoder_module(encoder, inputs)

        if isinstance(out, torch.Tensor):
            if out.dim() == 3:
                return out, out[:, 0, :]
            if out.dim() == 2:
                return None, out
            raise RuntimeError(f"Unsupported tensor output rank from encoder: {out.dim()}")

        if isinstance(out, Mapping):
            hidden = out.get("hidden_states")
            emb = out.get("embedding")
            if emb is None:
                raise RuntimeError("Encoder dict output must include key 'embedding'.")
            if not isinstance(emb, torch.Tensor) or emb.dim() != 2:
                raise RuntimeError("Encoder dict output 'embedding' must be a rank-2 tensor [B, D].")
            if hidden is not None and (not isinstance(hidden, torch.Tensor) or hidden.dim() != 3):
                raise RuntimeError("Encoder dict output 'hidden_states' must be rank-3 [B, L, D] or None.")
            return hidden, emb

        if isinstance(out, (tuple, list)):
            if len(out) != 2:
                raise RuntimeError("Encoder tuple output must be length 2: (hidden_or_none, embedding).")
            hidden, emb = out
            if not isinstance(hidden, torch.Tensor) and hidden is not None:
                raise RuntimeError("First tuple element from encoder must be a tensor or None.")
            if hidden is not None and hidden.dim() != 3:
                raise RuntimeError("First tuple element from encoder must be rank-3 hidden states or None.")
            if not isinstance(emb, torch.Tensor) or emb.dim() != 2:
                raise RuntimeError("Second tuple element from encoder must be rank-2 embeddings [B, D].")
            return hidden, emb

        raise RuntimeError(f"Unsupported encoder output type: {type(out)}")

    def encode_product(
        self,
        input_ids_or_inputs: Any,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        inputs = self._normalize_inputs(input_ids_or_inputs, attention_mask)
        hidden, emb = self._run_tower_encoder(self.product_encoder, inputs)
        return hidden, self.product_projector(emb)

    def encode_template(
        self,
        input_ids_or_inputs: Any,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        inputs = self._normalize_inputs(input_ids_or_inputs, attention_mask)
        hidden, emb = self._run_tower_encoder(self.template_encoder, inputs)
        return hidden, self.template_projector(emb)

    @staticmethod
    def _resolve_tower_inputs(
        inputs: Optional[Dict[str, torch.Tensor]],
        ids: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        name: str,
    ) -> Dict[str, torch.Tensor]:
        if inputs is not None:
            return inputs
        if ids is None or mask is None:
            raise RuntimeError(f"Either {name}_inputs or ({name}_ids, {name}_mask) must be provided.")
        return {"input_ids": ids, "attention_mask": mask}

    def _compute_mlm_logits(self, hidden: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.mlm_head is None or hidden is None:
            return None
        return self.mlm_head(hidden)

    def forward(
        self,
        product_ids: Optional[torch.Tensor] = None,
        product_mask: Optional[torch.Tensor] = None,
        template_ids: Optional[torch.Tensor] = None,
        template_mask: Optional[torch.Tensor] = None,
        product_inputs: Optional[Dict[str, torch.Tensor]] = None,
        template_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        product_inputs = self._resolve_tower_inputs(product_inputs, product_ids, product_mask, "product")
        template_inputs = self._resolve_tower_inputs(template_inputs, template_ids, template_mask, "template")

        prod_hidden, prod_cls = self.encode_product(product_inputs)
        templ_hidden, templ_cls = self.encode_template(template_inputs)
        if prod_cls.size(-1) != templ_cls.size(-1):
            raise RuntimeError(
                f"Product/template embedding dims mismatch: {prod_cls.size(-1)} vs {templ_cls.size(-1)}. "
                "Configure tower projection to a shared expected_dim."
            )

        prod_mlm = self._compute_mlm_logits(prod_hidden)
        templ_mlm = self._compute_mlm_logits(templ_hidden)

        return prod_cls, templ_cls, prod_mlm, templ_mlm


def infer_vocab_size(
    cfg: Dict[str, Any],
    tokenizer: Any,
    product_tower_cfg: Optional[Dict[str, Any]] = None,
    template_tower_cfg: Optional[Dict[str, Any]] = None,
) -> int:
    model_cfg = cfg.get("model", {})
    tok_cfg = cfg.get("tokenizer", {})
    if tokenizer is not None and hasattr(tokenizer, "vocab"):
        vocab_size = len(tokenizer.vocab)  # type: ignore[attr-defined]
    else:
        vocab_size = int(tok_cfg.get("vocab_size", 0))

    product_tower_cfg = product_tower_cfg or resolve_tower_cfg(cfg, "product")
    template_tower_cfg = template_tower_cfg or resolve_tower_cfg(cfg, "template")
    product_uses_default_bert = not bool(product_tower_cfg.get("encoder", {}).get("class_path"))
    template_uses_default_bert = not bool(template_tower_cfg.get("encoder", {}).get("class_path"))
    requires_vocab = (
        bool(model_cfg.get("mlm", {}).get("enabled", False)) or product_uses_default_bert or template_uses_default_bert
    )

    if vocab_size <= 0 and requires_vocab:
        raise RuntimeError(
            "Unable to determine tokenizer vocab size. Provide tokenizer.vocab_size for token-based encoders."
        )
    if vocab_size <= 0:
        vocab_size = 1
    return vocab_size


def build_model_from_config(cfg: Dict[str, Any], tokenizer: Any) -> ContrastiveModel:
    mcfg = cfg["model"]
    tcfg = cfg.get("tokenizer", {})
    product_tower_cfg = resolve_tower_cfg(cfg, "product")
    template_tower_cfg = resolve_tower_cfg(cfg, "template")
    vocab_size = infer_vocab_size(
        cfg, tokenizer, product_tower_cfg=product_tower_cfg, template_tower_cfg=template_tower_cfg
    )
    expected_dim = mcfg.get("expected_embedding_dim")
    if expected_dim is not None:
        expected_dim = int(expected_dim)

    return ContrastiveModel(
        vocab_size=vocab_size,
        hidden_size=mcfg["hidden_size"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        dropout=mcfg["dropout"],
        max_product_len=int(tcfg.get("max_product_len", 0)),
        max_template_len=int(tcfg.get("max_template_len", 0)),
        shared_encoder=mcfg["shared_encoder"],
        mlm_enabled=mcfg["mlm"]["enabled"],
        product_tower_cfg=product_tower_cfg,
        template_tower_cfg=template_tower_cfg,
        expected_embedding_dim=expected_dim,
    )
