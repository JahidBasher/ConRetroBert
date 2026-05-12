"""Template retrieval index for Stage-2 hard-negative sampling."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class Stage2RetrievalMixin:
    """Build and query a template retrieval index for Stage-2 negative sampling.

    Supports three backends:
    - ``faiss``: FAISS inner-product index (recommended for large template libraries).
    - ``matrix``: Dense CPU matmul (permitted only during validation to avoid O(N) cost).
    - ``none``: No retrieval; falls back to in-batch/random negatives only.

    This class is a standalone service injected into :class:`Stage2LightningModule`;
    the name is retained for backwards compatibility.

    Args:
        retrieval_cfg: Retrieval-specific config dict (the ``retrieval`` sub-dict
            of ``stage2.candidate_sampling``).
    """

    def __init__(self, retrieval_cfg: Dict[str, Any]) -> None:
        self._retrieval_cfg = retrieval_cfg
        self._backend: str = "none"
        self._faiss_index: Optional[Any] = None
        self._template_emb_cpu: Optional[torch.Tensor] = None

    def initialize(self, template_emb_cpu: torch.Tensor) -> None:
        """Build or load the retrieval index from normalized template embeddings.

        Must be called once (or after each embedding rebuild) before :meth:`retrieve`.

        Args:
            template_emb_cpu: CPU float32 tensor of shape ``(n_templates, embed_dim)``,
                L2-normalized.
        """
        self._template_emb_cpu = template_emb_cpu
        retrieval_cfg = self._retrieval_cfg

        if not retrieval_cfg.get("enabled", True):
            self._backend = "none"
            return

        backend = str(retrieval_cfg.get("backend", "faiss"))
        self._backend = backend
        if backend in ("none", "matrix"):
            return
        if backend != "faiss":
            raise RuntimeError(f"Unsupported Stage 2 retrieval backend: {backend!r}")

        try:
            import faiss  # type: ignore
        except Exception as exc:
            if retrieval_cfg.get("fallback_to_inbatch_random", True):
                self._backend = "none"
                print(
                    "FAISS is unavailable; Stage 2 retrieval-based hard negatives disabled "
                    "(using in-batch/random only)."
                )
                return
            raise RuntimeError(
                "FAISS is required for retrieval backend 'faiss' in Stage 2."
            ) from exc

        index = self._load_faiss_index(faiss, retrieval_cfg)
        if index is None:
            index = self._build_faiss_index(faiss, retrieval_cfg)
        nprobe = retrieval_cfg.get("nprobe")
        if nprobe is not None and hasattr(index, "nprobe"):
            index.nprobe = int(nprobe)
        self._faiss_index = index

    def retrieve(
        self, z_p_norm: torch.Tensor, top_k: int, allow_matrix: bool
    ) -> List[List[int]]:
        """Return top-k template IDs for each product embedding in the batch.

        Args:
            z_p_norm: L2-normalized product embeddings, shape ``(B, D)``.
            top_k: Maximum number of template IDs to return per sample.
            allow_matrix: Whether the ``matrix`` backend is permitted.  Should be
                False during training to avoid O(N) matmuls at every step.

        Returns:
            List of ``B`` integer lists, each containing up to ``top_k`` template IDs.
        """
        if top_k <= 0:
            return [[] for _ in range(z_p_norm.size(0))]

        k = min(top_k, self._template_emb_cpu.size(0))

        if self._backend == "faiss":
            return self._retrieve_faiss(z_p_norm, k)

        if self._backend == "matrix":
            if not allow_matrix:
                raise RuntimeError(
                    "Matrix retrieval is disabled in training to avoid O(260k) operations per step."
                )
            return self._retrieve_matrix(z_p_norm, k)

        # backend == "none": fall back to matrix when allowed (validation), otherwise empty
        if allow_matrix:
            return self._retrieve_matrix(z_p_norm, k)
        return [[] for _ in range(z_p_norm.size(0))]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_faiss_index(
        faiss_module: Any, retrieval_cfg: Dict[str, Any]
    ) -> Optional[Any]:
        """Load a prebuilt FAISS index when ``faiss_index_path`` exists on disk."""
        index_path = retrieval_cfg.get("faiss_index_path")
        if not index_path:
            return None
        path = Path(index_path)
        if not path.exists():
            return None
        return faiss_module.read_index(str(path))

    def _build_faiss_index(
        self, faiss_module: Any, retrieval_cfg: Dict[str, Any]
    ) -> Any:
        """Build an inner-product FAISS index from in-memory template embeddings."""
        embeddings_np = self._template_emb_cpu.float().numpy()
        index = faiss_module.IndexFlatIP(embeddings_np.shape[1])
        index.add(embeddings_np)
        save_path_str = retrieval_cfg.get("save_faiss_index_path")
        if save_path_str:
            save_path = Path(save_path_str)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            faiss_module.write_index(index, str(save_path))
        return index

    def _retrieve_matrix(self, z_p_norm: torch.Tensor, k: int) -> List[List[int]]:
        scores = z_p_norm.detach().cpu().float() @ self._template_emb_cpu.t()
        _, idx = torch.topk(scores, k=k, dim=-1)
        return idx.tolist()

    def _retrieve_faiss(self, z_p_norm: torch.Tensor, k: int) -> List[List[int]]:
        if self._faiss_index is None:
            raise RuntimeError("Stage 2 FAISS retrieval requested but index is missing.")
        query = z_p_norm.detach().cpu().float().numpy()
        _, idx = self._faiss_index.search(query, k)
        return idx.tolist()
