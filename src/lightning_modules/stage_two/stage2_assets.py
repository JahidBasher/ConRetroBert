"""Template embedding and token feature cache management for Stage-2 training."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from .datatypes import FeatureCache, TemplateEmbeddingConfig, TemplateCacheConfig


class Stage2TemplateAssetMixin:
    """Load, build, and cache template embeddings and token features for Stage-2.

    This class is a standalone service (injected into :class:`Stage2LightningModule`)
    rather than a mixin; the name is retained for backwards compatibility with any
    external code that references it.

    Args:
        model: Dual-encoder model exposing an ``encode_template(inputs)`` method.
        template_input_builder: Callable that tokenizes a single template SMARTS string
            into a feature dict.
        template_input_collator: Callable that batches a list of feature dicts into
            a single ``Dict[str, torch.Tensor]``.
        emb_cfg: Parsed template embedding cache configuration.
        token_cache_cfg: Parsed template token feature cache configuration.
    """

    def __init__(
        self,
        model: Any,
        template_input_builder: Callable[[str], Dict[str, Any]],
        template_input_collator: Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]],
        emb_cfg: TemplateEmbeddingConfig,
        token_cache_cfg: TemplateCacheConfig,
    ) -> None:
        self._model = model
        self._template_input_builder = template_input_builder
        self._template_input_collator = template_input_collator
        self._emb_cfg = emb_cfg
        self._token_cache_cfg = token_cache_cfg

    def load_or_build_embeddings(
        self,
        template_list: List[str],
        device: torch.device,
        force_rebuild: bool = False,
    ) -> torch.Tensor:
        """Return normalized template embeddings, loading from disk or encoding fresh.

        If a valid cache exists at ``emb_cfg.load_path`` and *force_rebuild* is False,
        the cached embeddings are loaded directly.  Otherwise, templates are encoded
        in batches by the model and the result is saved if ``emb_cfg.save_path`` is set.

        Args:
            template_list: Ordered list of SMARTS template strings.
            device: Device used for model inference during encoding.
            force_rebuild: When True, always re-encode even if a cache exists.

        Returns:
            CPU float32 tensor of shape ``(n_templates, embed_dim)``,
            L2-normalized along the embedding dimension.
        """
        load_path = self._emb_cfg.load_path
        if (not force_rebuild) and load_path and Path(load_path).exists():
            return self._load_cached_embeddings(load_path, template_list)

        feature_cache = self._load_feature_cache(template_list)
        template_emb = self._encode_embeddings(template_list, device, feature_cache)
        self.save_feature_cache_if_configured(template_list)
        self._save_embedding_cache(template_emb, template_list)
        return template_emb

    def save_feature_cache_if_configured(self, template_list: List[str]) -> None:
        """Persist template token features when saving is enabled.

        Keeps an existing cache file when it is already usable (or contains a
        valid embedding-only payload), and rebuilds otherwise.

        Args:
            template_list: Ordered list of SMARTS template strings to tokenize and save.
        """
        cache_path = self._token_cache_cfg.path
        if not (self._token_cache_cfg.save and cache_path):
            return
        cache_file = Path(cache_path)
        if cache_file.exists():
            should_keep_existing = False
            try:
                existing_payload = torch.load(cache_file, map_location="cpu")
                if hasattr(existing_payload, "get"):
                    self._validate_template_order(
                        existing_payload, template_list, "Template token cache"
                    )
                    existing_feats = self._extract_feature_tensors_from_payload(
                        existing_payload
                    )
                    should_keep_existing = (
                        existing_feats is not None
                        and self._is_usable_feature_cache(existing_feats, template_list)
                    )
                    if not should_keep_existing:
                        existing_emb = self._extract_embedding_tensor_from_payload(
                            existing_payload
                        )
                        should_keep_existing = (
                            isinstance(existing_emb, torch.Tensor)
                            and existing_emb.dim() == 2
                            and existing_emb.shape[0] == len(template_list)
                        )
            except RuntimeError:
                raise
            except Exception:
                should_keep_existing = False
            if should_keep_existing:
                return

        features = [self._template_input_builder(t) for t in template_list]
        feature_cache = self._template_input_collator(features)
        if not isinstance(feature_cache, dict):
            return

        Path(cache_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {"templates": template_list, "features": feature_cache}
        if "input_ids" in feature_cache:
            payload["input_ids"] = feature_cache["input_ids"]
        if "attention_mask" in feature_cache:
            payload["attention_mask"] = feature_cache["attention_mask"]
        torch.save(payload, cache_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_template_order(
        self, payload: Any, template_list: List[str], source: str
    ) -> None:
        """Validate cached template ordering against the active template library."""
        if not hasattr(payload, "get"):
            return
        templates = payload.get("templates")
        if templates is not None and templates != template_list:
            raise RuntimeError(f"{source} template order mismatch.")

    def _load_cached_embeddings(
        self, load_path: str, template_list: List[str]
    ) -> torch.Tensor:
        """Load and normalize cached template embeddings from disk."""
        payload = torch.load(load_path, map_location="cpu")
        self._validate_template_order(payload, template_list, "Template embedding cache")
        return F.normalize(payload["embeddings"].float(), dim=-1).contiguous()

    def _build_batch_inputs(
        self,
        start: int,
        batch_size: int,
        feature_cache: Optional[FeatureCache],
        template_list: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Build one template batch from cache slices or raw template text."""
        if feature_cache is not None:
            inputs = {
                k: v[start : start + batch_size].to(device=device)
                for k, v in feature_cache.items()
            }
            # Keep expected dtypes for token pipelines.
            if "input_ids" in inputs:
                inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.long)
            return inputs

        chunk = template_list[start : start + batch_size]
        features = [self._template_input_builder(t) for t in chunk]
        inputs = self._template_input_collator(features)
        if not isinstance(inputs, dict):
            raise RuntimeError("Template input collator must return a dict.")
        return {k: v.to(device=device) for k, v in inputs.items()}

    def _encode_embeddings(
        self,
        template_list: List[str],
        device: torch.device,
        feature_cache: Optional[FeatureCache],
    ) -> torch.Tensor:
        """Encode and L2-normalize all templates through the template tower."""
        outputs: List[torch.Tensor] = []
        was_training = self._model.training
        self._model.eval()
        for start in range(0, len(template_list), self._emb_cfg.encode_batch_size):
            inputs = self._build_batch_inputs(
                start, self._emb_cfg.encode_batch_size, feature_cache, template_list, device
            )
            with torch.no_grad():
                _, cls = self._model.encode_template(inputs)
                cls = F.normalize(cls, dim=-1)
            outputs.append(cls.cpu())
        if was_training:
            self._model.train()
        return torch.cat(outputs, dim=0).contiguous().float()

    def _save_embedding_cache(
        self, template_emb: torch.Tensor, template_list: List[str]
    ) -> None:
        """Persist normalized template embeddings when save_path is configured."""
        save_path = self._emb_cfg.save_path
        if not save_path:
            return
        Path(save_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        torch.save({"templates": template_list, "embeddings": template_emb}, save_path)

    @staticmethod
    def _extract_feature_tensors_from_payload(payload: Any) -> Optional[FeatureCache]:
        """Extract token feature tensors from known cache payload layouts."""
        if not hasattr(payload, "get"):
            return None

        candidate_dicts: List[Any] = []
        for key in ("features", "template_features"):
            nested = payload.get(key)
            if hasattr(nested, "items"):
                candidate_dicts.append(nested)
        candidate_dicts.append(payload)

        required_keys = ("input_ids", "attention_mask")
        for candidate in candidate_dicts:
            if not hasattr(candidate, "items"):
                continue
            feats = {
                key: value
                for key, value in candidate.items()
                if isinstance(value, torch.Tensor)
            }
            if all(key in feats for key in required_keys):
                return feats
        return None

    @staticmethod
    def _extract_embedding_tensor_from_payload(payload: Any) -> Optional[torch.Tensor]:
        """Extract an embedding matrix tensor from a cache payload when present."""
        if not hasattr(payload, "get"):
            return None
        emb = payload.get("embeddings")
        if isinstance(emb, torch.Tensor):
            return emb
        return None

    @staticmethod
    def _warn_invalid_template_token_cache(cache_path: str, reason: str) -> None:
        """Emit a non-fatal warning when a token cache payload cannot be consumed."""
        print(
            f"Stage 2 template token cache at '{cache_path}' is unusable ({reason}). "
            "Falling back to on-the-fly tokenization for this run."
        )

    @staticmethod
    def _is_usable_feature_cache(
        feature_cache: FeatureCache, template_list: List[str]
    ) -> bool:
        """Return True when cached token tensors are indexable and shape-aligned."""
        required_keys = ("input_ids", "attention_mask")
        n_templates = len(template_list)
        for key in required_keys:
            value = feature_cache.get(key)
            if not isinstance(value, torch.Tensor):
                return False
            if value.dim() < 1:
                return False
            if value.shape[0] != n_templates:
                return False
        return True

    def _load_feature_cache(self, template_list: List[str]) -> Optional[FeatureCache]:
        """Load cached template token features when configured and valid."""
        cache_path = self._token_cache_cfg.path
        if not (self._token_cache_cfg.load_if_exists and cache_path and Path(cache_path).exists()):
            return None
        payload = torch.load(cache_path, map_location="cpu")
        if not hasattr(payload, "get"):
            self._warn_invalid_template_token_cache(cache_path, "payload is not a dict")
            return None

        self._validate_template_order(payload, template_list, "Template token cache")
        feature_cache = self._extract_feature_tensors_from_payload(payload)
        if feature_cache is None:
            if self._extract_embedding_tensor_from_payload(payload) is not None:
                return None
            self._warn_invalid_template_token_cache(
                cache_path,
                "missing required token tensors (input_ids, attention_mask)",
            )
            return None
        if not self._is_usable_feature_cache(feature_cache, template_list):
            self._warn_invalid_template_token_cache(
                cache_path,
                "tensor shapes do not match current template library",
            )
            return None
        return feature_cache
