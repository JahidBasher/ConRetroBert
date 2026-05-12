"""Run one-step retrosynthesis ranking with template and reactant-set metrics."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import zipfile
import tqdm
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.data.validation import build_row_validator_from_config
from src.data.input_processing import collate_feature_dicts, get_text_input_builder
from src.model import build_model_from_config
from src.utils import load_config, load_or_build_tokenizer, load_weights_into_model


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class EvaluationCounters:
    exact_match_hits: Dict[int, int]
    unique_reactant_set_count_sum: Dict[int, int]
    unique_reactant_set_count_denoms: Dict[int, int]
    template_success_hits: Dict[int, int]
    template_success_denoms: Dict[int, int]
    template_retrieval_hits: Dict[int, int]
    gt_template_yield_hits: Dict[int, int]
    gt_template_yield_denoms: Dict[int, int]
    gt_template_yield_coverage_hits: Dict[int, int]
    rows_with_valid_ground_truth: int = 0
    rows_with_valid_ground_truth_template: int = 0
    rows_with_predictions: int = 0
    template_attempts_total: int = 0
    template_success_total: int = 0

    @classmethod
    def create(cls, top_k_values: Sequence[int]) -> "EvaluationCounters":
        """Initialise all per-k counters to zero for the given k list."""
        return cls(
            exact_match_hits={k: 0 for k in top_k_values},
            unique_reactant_set_count_sum={k: 0 for k in top_k_values},
            unique_reactant_set_count_denoms={k: 0 for k in top_k_values},
            template_success_hits={k: 0 for k in top_k_values},
            template_success_denoms={k: 0 for k in top_k_values},
            template_retrieval_hits={k: 0 for k in top_k_values},
            gt_template_yield_hits={k: 0 for k in top_k_values},
            gt_template_yield_denoms={k: 0 for k in top_k_values},
            gt_template_yield_coverage_hits={k: 0 for k in top_k_values},
        )


@dataclass
class EvalConfig:
    """Evaluation hyper-parameters extracted from CLI args.

    Decouples inner evaluation functions from the raw argparse.Namespace so
    they can be tested and called without constructing a full argument parser.

    Attributes:
        template_top_k: Number of top templates retrieved per product.
        max_outcomes_per_template: Maximum RDKit outcomes kept per template.
        max_reactant_sets: Maximum unique reactant sets ranked per product.
        batch_size: Products per retrieval batch.
        summary: Whether to print and/or write a summary JSON.
        summary_out: Path to write the summary JSON file (None = skip).
        predictions_out: Path to write ranked predictions JSONL (None = skip).
        break_on_n: Stop after this many input rows (debug / smoke-test).
    """

    template_top_k: int
    max_outcomes_per_template: int
    max_reactant_sets: int
    batch_size: int
    summary: bool
    summary_out: Optional[str]
    predictions_out: Optional[str]
    break_on_n: Optional[int]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EvalConfig":
        """Build from a parsed argparse.Namespace."""
        return cls(
            template_top_k=args.template_top_k,
            max_outcomes_per_template=args.max_outcomes_per_template,
            max_reactant_sets=args.max_reactant_sets,
            batch_size=args.batch_size,
            summary=args.summary,
            summary_out=args.summary_out,
            predictions_out=args.predictions_out,
            break_on_n=args.break_on_n,
        )


@dataclass
class RDKitOps:
    """Lazily loaded RDKit template application callables.

    Loading is attempted once via :meth:`load`; a missing or broken RDKit
    installation results in ``None`` callables without raising, so the
    evaluation can still run in template-retrieval-only mode.

    Attributes:
        apply_template: Callable ``(product_smi, template_smarts) ->
            Iterable[Iterable[str]]``  — yields reactant-set tuples.
    """

    apply_template: Optional[Callable]

    @property
    def available(self) -> bool:
        """True when template application is functional."""
        return callable(self.apply_template)

    @classmethod
    def load(cls) -> "RDKitOps":
        """Attempt to import rdkit_utils and return a populated instance."""
        try:
            from src.rdkit_utils import apply_template
            return cls(apply_template=apply_template)
        except Exception:
            return cls(apply_template=None)


@dataclass
class ArtifactPaths:
    """Resolved output paths for one evaluation run.

    Attributes:
        stage: Training stage inferred from the experiment config.
        artifact_dir: Root directory for all run artifacts.
        config_snapshot: Path to the saved config JSON.
        code_snapshot: Path to the saved code tar.gz, or None on failure.
    """

    stage: int
    artifact_dir: Path
    config_snapshot: Path
    code_snapshot: Optional[Path]


@dataclass
class RetrievalContext:
    """Fixed retrieval state shared across all batches in one evaluation run.

    Attributes:
        model: Inference model with ``encode_product`` method.
        product_input_builder: Tokenises a single product SMILES string.
        templates: Ordered template library strings.
        template_embeddings_device: Pre-loaded embedding matrix on the target
            device, or None when using FAISS.
        faiss_index: Loaded FAISS index, or None when using dense similarity.
        model_device: Device for model inference.
        use_fp16_similarity: Whether to cast product embeddings to fp16 for
            the similarity dot product.
    """

    model: Any
    product_input_builder: Callable
    templates: List[str]
    template_embeddings_device: Optional[torch.Tensor]
    faiss_index: Any
    model_device: torch.device
    use_fp16_similarity: bool


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse CLI configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--templates")
    parser.add_argument("--cache", help="Template cache .pt from build_template_cache.py")
    parser.add_argument("--eval_jsonl", required=True, help="JSONL with product and reactants fields")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Products per retrieval batch for inference (default: 1)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0, ... (default: auto)",
    )
    parser.add_argument("--template-top-k", "--top_k", dest="template_top_k", type=int, default=50)
    parser.add_argument("--max-outcomes-per-template", type=int, default=4)
    parser.add_argument("--max-reactant-sets", type=int, default=100)
    parser.add_argument("--eval-k", default="1,3,5,10", help="Comma-separated k values")
    parser.add_argument("--summary", action="store_true", help="Print evaluation summary")
    parser.add_argument("--summary-out", "--summary_out", dest="summary_out", default=None)
    parser.add_argument("--predictions-out", "--predictions_out", dest="predictions_out", default=None)
    parser.add_argument(
        "--save-top-max-k-templates",
        action="store_true",
        help=(
            "If set, save per-product top-max-k retrieved templates "
            "(k = max(--eval-k)) into artifact_dir/top_max_k_templates.jsonl."
        ),
    )
    parser.add_argument("--break_on_n", type=int, default=None)
    parser.add_argument(
        "--rebuild_cache",
        action="store_true",
        help="Force rebuild the template cache for this run instead of reusing an existing .pt cache.",
    )
    parser.add_argument("--faiss_index", default=None)
    parser.add_argument("--faiss_nprobe", type=int, default=None)
    parser.add_argument("--faiss_ef_search", type=int, default=None)
    parser.add_argument("--artifact_dir", "--artifact-dir", default=None)
    parser.add_argument("--artifact_name", "--artifact-name", default=None)
    parser.add_argument(
        "--save_code_snapshot",
        action="store_true",
        help="Save a zip snapshot of the current codebase into the artifact directory.",
    )
    return parser.parse_args()


def _build_arg_snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    """Return a compact record of CLI settings used for this run."""
    return {
        "command": " ".join(sys.argv),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "templates": args.templates,
        "cache": args.cache,
        "rebuild_cache": args.rebuild_cache,
        "eval_jsonl": args.eval_jsonl,
        "batch_size": args.batch_size,
        "device": args.device,
        "template_top_k": args.template_top_k,
        "max_outcomes_per_template": args.max_outcomes_per_template,
        "max_reactant_sets": args.max_reactant_sets,
        "eval_k": args.eval_k,
        "faiss_index": args.faiss_index,
        "faiss_nprobe": args.faiss_nprobe,
        "faiss_ef_search": args.faiss_ef_search,
        "artifact_dir": args.artifact_dir,
        "artifact_name": args.artifact_name,
        "save_code_snapshot": args.save_code_snapshot,
        "summary_out": args.summary_out,
        "predictions_out": args.predictions_out,
        "save_top_max_k_templates": args.save_top_max_k_templates,
        "break_on_n": args.break_on_n,
    }


def _validate_args(args: argparse.Namespace) -> None:
    """Raise on clearly invalid argument combinations before any I/O."""
    if args.max_outcomes_per_template < 0:
        raise RuntimeError("--max-outcomes-per-template must be >= 0")
    if args.max_reactant_sets <= 0:
        raise RuntimeError("--max-reactant-sets must be > 0")
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def _save_tracked_code_snapshot(snapshot_path: Path) -> Optional[Path]:
    """Archive the full project root into a zip at *snapshot_path*."""
    root_path = Path(__file__).resolve().parent
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            snapshot_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for fp in sorted(root_path.rglob("*")):
                if fp == snapshot_path or not fp.is_file():
                    continue
                rel = fp.relative_to(root_path)
                arcname = (Path(root_path.name) / rel).as_posix()
                archive.write(fp, arcname=arcname)
    except Exception:
        try:
            snapshot_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    return snapshot_path


def _setup_artifacts(
    config: Dict[str, Any],
    checkpoint_path: str,
    artifact_dir_override: Optional[str],
    artifact_name: Optional[str],
    save_code_snapshot: bool,
) -> ArtifactPaths:
    """Create the run artifact directory, save config snapshot and code archive.

    The artifact root is resolved from (in priority order): the CLI
    ``--artifact_dir`` override, ``config.checkpoint.dirpath``,
    ``config.training.output_dir``, or a heuristic based on the checkpoint path.

    Args:
        config: Full experiment config dict.
        checkpoint_path: Path to the model checkpoint being evaluated.
        artifact_dir_override: Explicit root override from CLI (may be None).
        artifact_name: Optional named sub-directory under the root.
        save_code_snapshot: Whether to save a zip snapshot of the codebase.

    Returns:
        :class:`ArtifactPaths` with all resolved output locations.
    """
    try:
        stage = int(config.get("training", {}).get("stage", 1))
    except Exception:
        stage = 1

    if artifact_dir_override:
        artifact_root = Path(artifact_dir_override).expanduser()
    else:
        checkpoint_cfg = config.get("checkpoint", {}).get("dirpath")
        output_dir = config.get("training", {}).get("output_dir")
        if checkpoint_cfg:
            artifact_root = Path(checkpoint_cfg)
        elif output_dir:
            artifact_root = Path(output_dir) / f"checkpoints_stage_{stage}"
        else:
            resolved_checkpoint = Path(checkpoint_path).resolve()
            artifact_root = Path("artifacts") / f"checkpoints_stage_{stage}"
            for parent in (resolved_checkpoint, *resolved_checkpoint.parents):
                if "checkpoints_stage_" in parent.name:
                    artifact_root = parent
                    break

    resolved_checkpoint = Path(checkpoint_path).resolve()
    stem = resolved_checkpoint.stem
    if artifact_name:
        artifact_root.mkdir(parents=True, exist_ok=True)
        artifact_dir = artifact_root / artifact_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        config_snapshot = artifact_dir / "config.json"
        code_snapshot = (
            _save_tracked_code_snapshot(
                artifact_dir / f"code_snapshot_{datetime.datetime.now():%Y%m%d_%H%M%S}.zip"
            )
            if save_code_snapshot
            else None
        )
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_dir = resolved_checkpoint.parent / f"{stem}_{timestamp}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        config_snapshot = artifact_dir / "config.json"
        code_snapshot = (
            _save_tracked_code_snapshot(
                artifact_dir / f"code_snapshot_{datetime.datetime.now():%Y%m%d_%H%M%S}.zip"
            )
            if save_code_snapshot
            else None
        )

    with open(config_snapshot, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    return ArtifactPaths(
        stage=stage,
        artifact_dir=artifact_dir,
        config_snapshot=config_snapshot,
        code_snapshot=code_snapshot,
    )


# ---------------------------------------------------------------------------
# Device + model
# ---------------------------------------------------------------------------


def _resolve_eval_device(raw_device: str) -> torch.device:
    """Resolve CLI device string into a :class:`torch.device` with validation.

    Args:
        raw_device: Raw ``--device`` CLI value (e.g. ``"auto"``, ``"cuda:0"``).

    Returns:
        Validated :class:`torch.device`.
    """
    resolved = str(raw_device or "auto").strip()
    if resolved.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(resolved)
    except Exception as exc:
        raise RuntimeError(f"Invalid --device value: {raw_device}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but torch.cuda.is_available() is False.")
    return device


def _load_model(
    config: Dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
) -> Any:
    """Build model from config, load checkpoint weights, move to device.

    Args:
        config: Full experiment config dict.
        checkpoint_path: Path to the checkpoint containing model weights.
        device: Target inference device.

    Returns:
        Model in eval mode on *device*.
    """
    tokenizer = load_or_build_tokenizer(config, allow_build=False)
    model = build_model_from_config(config, tokenizer)
    load_weights_into_model(model, checkpoint_path)
    model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Template loading + FAISS
# ---------------------------------------------------------------------------


def _load_template_embeddings(
    cache_path: Optional[str],
    templates_path: Optional[str],
    checkpoint_path: str,
    config_path: str,
    rebuild_cache: bool = False,
) -> Tuple[List[str], Optional[torch.Tensor], bool]:
    """Load templates and their embeddings from cache or build via subprocess.

    If no cache path is given, the cache is built next to the checkpoint for
    later reuse.  Returns ``(templates, embeddings, cache_fp16)``.

    Args:
        cache_path: Path to a prebuilt ``.pt`` template cache (may be None).
        templates_path: Path to the template JSONL source (may be None).
        checkpoint_path: Path to the model checkpoint (used for cache naming).
        config_path: Path to the experiment config (passed to cache builder).
        rebuild_cache: When True, ignore any existing cache file and rebuild it.

    Returns:
        Tuple of template strings, CPU embedding tensor (or None), and a flag
        indicating whether the cached embeddings are stored as fp16.
    """
    def _read_cache(path: str) -> Tuple[List[str], Optional[torch.Tensor], bool]:
        payload = torch.load(path, map_location="cpu")
        emb = payload.get("embeddings")
        return payload["templates"], emb, bool(emb is not None and emb.dtype == torch.float16)

    if not templates_path:
        if rebuild_cache:
            raise RuntimeError(
                "--rebuild_cache requires --templates so the template cache can be regenerated."
            )
        if cache_path:
            return _read_cache(cache_path)
        raise RuntimeError("Provide --cache or --templates for template library.")

    resolved_ckpt = Path(checkpoint_path).resolve()
    resolved_cache = Path(cache_path).resolve() if cache_path else (
        resolved_ckpt.parent / f"{resolved_ckpt.stem}.template_cache.pt"
    )

    if resolved_cache.exists() and not rebuild_cache:
        print(f"[template cache] loading existing cache from {resolved_cache}")
        return _read_cache(str(resolved_cache))

    action = "rebuilding" if rebuild_cache and resolved_cache.exists() else "generating"
    print(
        f"[template cache] {action} cache at {resolved_cache} "
        "(batch_size=1024, device=cuda, fp16, num_workers=16)"
    )
    repo_root = Path(__file__).resolve().parent
    cache_builder = repo_root / "scripts" / "build_template_cache.py"
    os.makedirs(resolved_cache.parent, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run(
        [
            sys.executable, str(cache_builder),
            "--config", config_path,
            "--checkpoint", checkpoint_path,
            "--templates", templates_path,
            "--output", str(resolved_cache),
            "--batch_size", "1024",
            "--device", "cuda",
            "--fp16",
            "--num_workers", "16",
        ],
        check=True, cwd=repo_root, env=env,
    )
    return _read_cache(str(resolved_cache))


def _load_faiss_index(
    faiss_index_path: Optional[str],
    faiss_nprobe: Optional[int],
    faiss_ef_search: Optional[int],
) -> Any:
    """Load and configure a FAISS index from disk.

    Args:
        faiss_index_path: Path to the FAISS index file (None = no index).
        faiss_nprobe: IVF nprobe parameter (None = use index default).
        faiss_ef_search: HNSW efSearch parameter (None = use index default).

    Returns:
        Configured FAISS index, or None if *faiss_index_path* is None.
    """
    if not faiss_index_path:
        return None

    try:
        import faiss
    except Exception as exc:
        raise RuntimeError("faiss is not installed; cannot use FAISS index.") from exc

    index = faiss.read_index(faiss_index_path)
    if faiss_nprobe is not None and hasattr(index, "nprobe"):
        index.nprobe = faiss_nprobe
    if faiss_ef_search is not None and hasattr(index, "hnsw"):
        index.hnsw.efSearch = faiss_ef_search
    return index


# ---------------------------------------------------------------------------
# SMILES / ground truth helpers
# ---------------------------------------------------------------------------


def _normalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES and strip atom mapping numbers."""
    try:
        from rdkit import Chem
    except Exception:
        return None

    if not smiles or not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _make_reactant_key(reactants: Sequence[str]) -> Optional[str]:
    """Build a deterministic reactant-set key from canonicalized components."""
    if not reactants:
        return None
    normalized = [_normalize_smiles(r) for r in reactants]
    if any(s is None for s in normalized):
        return None
    return ".".join(sorted(normalized))  # type: ignore[arg-type]


def _collect_ground_truth_signature(obj: Dict[str, Any]) -> Optional[str]:
    """Extract canonical ground-truth reactant-set signature from a dataset row."""
    raw = obj.get("reactants", "")
    if isinstance(raw, list):
        parts = [str(x).strip() for x in raw if isinstance(x, str)]
    elif isinstance(raw, str):
        parts = [p.strip() for p in raw.split(".") if p.strip()]
    else:
        return None
    return _make_reactant_key(tuple(parts)) if parts else None


def _collect_ground_truth_templates(obj: Dict[str, Any]) -> List[str]:
    """Extract one or more ground-truth template SMARTS from a dataset row."""
    template_fields = (
        "template", "template_smarts", "reaction_template",
        "retro_template", "gt_template", "gt_templates", "templates",
    )
    templates: List[str] = []
    for field in template_fields:
        raw = obj.get(field)
        if isinstance(raw, str) and raw.strip():
            templates.append(raw.strip())
        elif isinstance(raw, list):
            templates.extend(v.strip() for v in raw if isinstance(v, str) and v.strip())
    return sorted(set(templates))


# ---------------------------------------------------------------------------
# Retrieval + ranking
# ---------------------------------------------------------------------------


def _retrieve_template_candidates_batch(
    product_smiles_batch: Sequence[str],
    retrieval_ctx: RetrievalContext,
    template_top_k: int,
) -> List[List[Dict[str, float]]]:
    """Retrieve top-k templates for each product in a batch.

    Uses FAISS when a FAISS index is present; falls back to dense dot-product
    similarity over the preloaded embedding matrix.

    Args:
        product_smiles_batch: Product SMILES strings for this batch.
        retrieval_ctx: Fixed retrieval state (model, embeddings, device, etc.).
        template_top_k: Number of top templates to retrieve per product.

    Returns:
        Per-product list of ``{"template": str, "score": float}`` dicts,
        ordered by descending score.
    """
    if not product_smiles_batch:
        return []

    ctx = retrieval_ctx
    prod_inputs = collate_feature_dicts([ctx.product_input_builder(s) for s in product_smiles_batch])
    prod_inputs = {
        k: v.to(device=ctx.model_device, non_blocking=(ctx.model_device.type == "cuda"))
        if torch.is_tensor(v) else v
        for k, v in prod_inputs.items()
    }
    with torch.no_grad():
        _, prod_cls = ctx.model.encode_product(prod_inputs)
        prod_cls = F.normalize(prod_cls, dim=-1)

    if ctx.faiss_index is not None:
        q = prod_cls.cpu().float().numpy()
        k = min(template_top_k, ctx.faiss_index.ntotal)
        if k <= 0:
            return [[] for _ in product_smiles_batch]
        vals, idx = ctx.faiss_index.search(q, k)
        vals = torch.tensor(vals)
        idx = torch.tensor(idx)
    else:
        if ctx.template_embeddings_device is None:
            raise RuntimeError("Template embeddings not available. Provide --cache or --faiss_index.")
        query = prod_cls.half() if ctx.use_fp16_similarity else prod_cls
        if query.dtype != ctx.template_embeddings_device.dtype:
            query = query.to(dtype=ctx.template_embeddings_device.dtype)
        if query.device.type == "cpu" and query.dtype == torch.float16:
            query = query.float()
        score_matrix = query @ ctx.template_embeddings_device.t()
        k = min(template_top_k, score_matrix.size(1))
        if k <= 0:
            return [[] for _ in product_smiles_batch]
        vals, idx = torch.topk(score_matrix, k=k, dim=-1)

    return [
        [{"template": ctx.templates[j], "score": float(s)} for s, j in zip(row_scores, row_idx)]
        for row_scores, row_idx in zip(vals.tolist(), idx.tolist())
    ]


def _rank_reactant_set_candidates(
    product_smiles: str,
    template_results: List[Dict[str, float]],
    max_outcomes_per_template: int,
    rdkit: RDKitOps,
) -> Tuple[List[Dict[str, Any]], int, int, List[int], List[List[str]]]:
    """Create and rank unique reactant-set candidates from template outcomes.

    Applies each retrieved template to the product SMILES via RDKit, deduplicates
    reactant sets (keeping the highest-scoring template), and returns them sorted
    by descending score.

    Args:
        product_smiles: The query product SMILES.
        template_results: Retrieved templates with scores.
        max_outcomes_per_template: Cap on RDKit outcomes per template (0 = unlimited).
        rdkit: Loaded RDKit ops; if unavailable, returns empty candidates.

    Returns:
        ``(ranked, attempts, successes, success_mask, per_template_outcome_keys)``
    """
    if not rdkit.available:
        return [], 0, 0, [], [[] for _ in template_results]

    candidates: Dict[str, Tuple[float, str, Tuple[str, ...]]] = {}
    per_template_outcome_keys: List[List[str]] = []
    attempts = 0
    template_success = 0
    template_success_mask: List[int] = []

    for item in template_results:
        attempts += 1
        template = item.get("template", "")
        score = float(item.get("score", 0.0))

        reactant_tuples: List[Tuple[str, ...]] = []
        template_outcome_keys: List[str] = []
        for reactant_set in rdkit.apply_template(product_smiles, template):  # type: ignore[misc]
            reactant_tuples.append(tuple(reactant_set))
            if max_outcomes_per_template > 0 and len(reactant_tuples) >= max_outcomes_per_template:
                break

        if reactant_tuples:
            template_success += 1
        template_success_mask.append(1 if reactant_tuples else 0)

        for reactants in reactant_tuples:
            key = _make_reactant_key(reactants)
            if key is None:
                continue
            template_outcome_keys.append(key)
            prev = candidates.get(key)
            if prev is None or score > prev[0]:
                candidates[key] = (score, template, tuple(reactants))
        per_template_outcome_keys.append(template_outcome_keys)

    ranked = [
        {"reactants": key, "score": p[0], "template": p[1], "template_reactants": p[2]}
        for key, p in sorted(candidates.items(), key=lambda kv: (-kv[1][0], kv[0]))
    ]
    return (
        ranked,
        attempts,
        template_success,
        template_success_mask,
        per_template_outcome_keys,
    )


def _evaluate_product_row(
    row: Dict[str, Any],
    product_smiles: str,
    template_candidates: List[Dict[str, float]],
    eval_cfg: EvalConfig,
    top_k_values: Sequence[int],
    rdkit: RDKitOps,
    counters: EvaluationCounters,
) -> Dict[str, Any]:
    """Process one product row after retrieval and update evaluation counters.

    Args:
        row: Raw dataset row dict.
        product_smiles: Product SMILES extracted from *row*.
        template_candidates: Retrieved templates with scores for this product.
        eval_cfg: Evaluation configuration.
        top_k_values: Sorted list of k values for metric reporting.
        rdkit: Loaded RDKit ops.
        counters: Mutable counters updated in place.

    Returns:
        Output dict for this row, suitable for JSONL serialization.
    """
    retrieved_templates = [item["template"] for item in template_candidates]
    gt_templates = _collect_ground_truth_templates(row)
    gt_template_set = set(gt_templates)
    gt_signature = _collect_ground_truth_signature(row)

    (
        ranked,
        template_attempts,
        template_success,
        success_mask,
        per_template_outcome_keys,
    ) = _rank_reactant_set_candidates(
        product_smiles=product_smiles,
        template_results=template_candidates,
        max_outcomes_per_template=eval_cfg.max_outcomes_per_template,
        rdkit=rdkit,
    )
    counters.template_attempts_total += template_attempts
    counters.template_success_total += template_success

    top_ranked = ranked[: eval_cfg.max_reactant_sets]
    ranked_rows = [
        {
            "rank": rank,
            "score": item["score"],
            "reactant_set_smiles": item["reactants"],
            "template_smarts": item["template"],
            "template_reactant_set_smiles": list(item["template_reactants"]),
        }
        for rank, item in enumerate(top_ranked, start=1)
    ]

    for k in top_k_values:
        cutoff = min(k, len(template_candidates))
        counters.template_success_denoms[k] += cutoff
        counters.template_success_hits[k] += sum(success_mask[:cutoff])
        if cutoff > 0:
            unique_reactant_sets = len(
                {
                    reactant_key
                    for template_keys in per_template_outcome_keys[:cutoff]
                    for reactant_key in template_keys
                }
            )
            counters.unique_reactant_set_count_sum[k] += unique_reactant_sets
            counters.unique_reactant_set_count_denoms[k] += 1
        if gt_template_set and any(t in gt_template_set for t in retrieved_templates[:cutoff]):
            counters.template_retrieval_hits[k] += 1

    row_out: Dict[str, Any] = {
        "product_smiles": product_smiles,
        "ranked_reactant_set_predictions": ranked_rows,
        "template_attempts": template_attempts,
        "template_successes": template_success,
    }
    if gt_templates:
        counters.rows_with_valid_ground_truth_template += 1
        row_out["ground_truth_template_smarts"] = gt_templates

    if gt_signature is not None:
        counters.rows_with_valid_ground_truth += 1
        row_out["ground_truth_reactant_set"] = gt_signature
        ranked_keys = [item["reactants"] for item in top_ranked]
        for k in top_k_values:
            cutoff = min(k, len(ranked_keys))
            if gt_signature in ranked_keys[:cutoff]:
                counters.exact_match_hits[k] += 1
            template_cutoff = min(k, len(per_template_outcome_keys))
            per_template_hits = sum(
                1 for keys in per_template_outcome_keys[:template_cutoff] if gt_signature in set(keys)
            )
            counters.gt_template_yield_hits[k] += per_template_hits
            counters.gt_template_yield_denoms[k] += template_cutoff
            if per_template_hits > 0:
                counters.gt_template_yield_coverage_hits[k] += 1

    if top_ranked:
        counters.rows_with_predictions += 1

    return row_out


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def _process_batch(
    batch_rows: List[Dict[str, Any]],
    batch_products: List[str],
    retrieval_ctx: RetrievalContext,
    eval_cfg: EvalConfig,
    top_k_values: Sequence[int],
    rdkit: RDKitOps,
    counters: EvaluationCounters,
    predictions_file: Any,
    top_templates_file: Any,
    top_templates_k: int,
) -> None:
    """Retrieve candidates for a product batch and evaluate each row.

    Args:
        batch_rows: Raw dataset rows for this batch.
        batch_products: Product SMILES aligned with *batch_rows*.
        retrieval_ctx: Fixed retrieval state shared across all batches.
        eval_cfg: Evaluation configuration.
        top_k_values: Sorted k values for metric reporting.
        rdkit: Loaded RDKit ops.
        counters: Mutable counters updated in place.
        predictions_file: Open file handle for predictions JSONL, or None.
        top_templates_file: Open file handle for per-product top-template dump JSONL, or None.
        top_templates_k: Number of top templates to dump per product.
    """
    if not batch_rows:
        return

    candidates_batch = _retrieve_template_candidates_batch(
        product_smiles_batch=batch_products,
        retrieval_ctx=retrieval_ctx,
        template_top_k=eval_cfg.template_top_k,
    )
    for row, product_smiles, candidates in zip(batch_rows, batch_products, candidates_batch):
        if top_templates_file is not None:
            top_templates_row = {
                "product_smiles": product_smiles,
                "top_k": top_templates_k,
                "top_templates": [
                    {
                        "rank": rank,
                        "template_smarts": item.get("template", ""),
                        "score": float(item.get("score", 0.0)),
                    }
                    for rank, item in enumerate(candidates[:top_templates_k], start=1)
                ],
            }
            if isinstance(row, dict):
                if "id" in row:
                    top_templates_row["id"] = row.get("id")
                if "split" in row:
                    top_templates_row["split"] = row.get("split")
            top_templates_file.write(json.dumps(top_templates_row) + "\n")

        row_out = _evaluate_product_row(
            row=row,
            product_smiles=product_smiles,
            template_candidates=candidates,
            eval_cfg=eval_cfg,
            top_k_values=top_k_values,
            rdkit=rdkit,
            counters=counters,
        )
        if predictions_file is not None:
            predictions_file.write(json.dumps(row_out) + "\n")


def _run_evaluation_loop(
    eval_jsonl: str,
    retrieval_ctx: RetrievalContext,
    eval_cfg: EvalConfig,
    top_k_values: Sequence[int],
    row_validator: Any,
    rdkit: RDKitOps,
    top_templates_out: Optional[str],
    top_templates_k: int,
) -> Tuple[EvaluationCounters, Dict[str, int]]:
    """Stream through the evaluation JSONL and accumulate all metrics.

    Args:
        eval_jsonl: Path to the evaluation dataset JSONL.
        retrieval_ctx: Fixed retrieval state.
        eval_cfg: Evaluation configuration.
        top_k_values: Sorted k values for metric reporting.
        row_validator: Optional row filter from config (None = no filtering).
        rdkit: Loaded RDKit ops.
        top_templates_out: Optional path for per-product top-template dump JSONL.
        top_templates_k: Number of templates to dump per product.

    Returns:
        ``(counters, filter_reason_counts)`` — accumulated metric counters and
        a dict mapping filter rejection reasons to their counts.
    """
    counters = EvaluationCounters.create(top_k_values)
    filter_reason_counts: Dict[str, int] = defaultdict(int)
    filter_seen = filter_kept = filter_dropped = 0

    predictions_file = None
    top_templates_file = None
    if eval_cfg.predictions_out:
        predictions_file = open(eval_cfg.predictions_out, "w", encoding="utf-8")
    if top_templates_out:
        top_templates_file = open(top_templates_out, "w", encoding="utf-8")

    try:
        pending_rows: List[Dict[str, Any]] = []
        pending_products: List[str] = []

        with open(eval_jsonl, "r", encoding="utf-8") as fh:
            for index, line in enumerate(tqdm.tqdm(fh), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue

                if eval_cfg.break_on_n and index == eval_cfg.break_on_n:
                    print(f"[eval] stopping after {eval_cfg.break_on_n} rows (--break_on_n)")
                    break

                product_smiles = row.get("product", "")
                if not product_smiles or not isinstance(product_smiles, str):
                    continue

                if row_validator is not None:
                    filter_seen += 1
                    keep, reason = row_validator(row)
                    if not keep:
                        filter_dropped += 1
                        filter_reason_counts[str(reason or "filtered")] += 1
                        continue
                    filter_kept += 1

                pending_rows.append(row)
                pending_products.append(product_smiles)
                if len(pending_rows) >= eval_cfg.batch_size:
                    _process_batch(
                        pending_rows, pending_products, retrieval_ctx,
                        eval_cfg, top_k_values, rdkit,
                        counters, predictions_file, top_templates_file, top_templates_k,
                    )
                    pending_rows.clear()
                    pending_products.clear()

        if pending_rows:
            _process_batch(
                pending_rows, pending_products, retrieval_ctx,
                eval_cfg, top_k_values, rdkit,
                counters, predictions_file, top_templates_file, top_templates_k,
            )
    finally:
        if predictions_file is not None:
            predictions_file.close()
        if top_templates_file is not None:
            top_templates_file.close()

    if row_validator is not None:
        print(
            f"[filter] seen={filter_seen} kept={filter_kept} "
            f"dropped={filter_dropped} reasons={dict(filter_reason_counts)}"
        )

    return counters, dict(filter_reason_counts)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _safe_pct(num: int, den: int) -> float:
    """Return ``num / den`` as a float, or 0.0 when *den* is zero."""
    return float(num) / float(den) if den > 0 else 0.0


def _build_summary(
    counters: EvaluationCounters,
    top_k_values: Sequence[int],
    artifact: ArtifactPaths,
    arg_snapshot: Dict[str, Any],
    filter_reason_counts: Dict[str, int],
    row_validator_enabled: bool,
) -> Dict[str, Any]:
    """Build the evaluation summary dict from accumulated counters.

    Args:
        counters: Final accumulated evaluation counters.
        top_k_values: Sorted k values.
        artifact: Artifact path info for the run metadata block.
        arg_snapshot: CLI arg snapshot dict for the run metadata block.
        filter_reason_counts: Per-reason counts from the validation filter.
        row_validator_enabled: Whether a row validator was active.

    Returns:
        Nested summary dict suitable for JSON serialization.
    """
    pct = _safe_pct

    exact_rates = {
        f"top{k}_accuracy": pct(counters.exact_match_hits[k], counters.rows_with_valid_ground_truth)
        for k in top_k_values
    }
    template_retrieval_rates = {
        f"template_top{k}_accuracy": pct(
            counters.template_retrieval_hits[k],
            counters.rows_with_valid_ground_truth_template,
        )
        for k in top_k_values
    }
    template_applicability_rates = {
        f"template_applicability_rate@{k}": pct(
            counters.template_success_hits[k], counters.template_success_denoms[k]
        )
        for k in top_k_values
    }
    reactant_set_diversity_rates = {
        f"mean_unique_reactant_sets_per_product@{k}": pct(
            counters.unique_reactant_set_count_sum[k],
            counters.unique_reactant_set_count_denoms[k],
        )
        for k in top_k_values
    }
    gt_template_yield_count_rates = {
        f"gt_template_yield_count@{k}": pct(
            counters.gt_template_yield_hits[k],
            counters.rows_with_valid_ground_truth,
        )
        for k in top_k_values
    }
    gt_template_yield_rate_rates = {
        f"gt_template_yield_rate@{k}": pct(
            counters.gt_template_yield_hits[k],
            counters.gt_template_yield_denoms[k],
        )
        for k in top_k_values
    }
    gt_template_yield_coverage_rates = {
        f"gt_template_yield_coverage@{k}": pct(
            counters.gt_template_yield_coverage_hits[k],
            counters.rows_with_valid_ground_truth,
        )
        for k in top_k_values
    }
    metric_definitions: Dict[str, str] = {
        "rows_with_predictions": "Number of rows that produced at least one ranked reactant-set prediction.",
        "total": "Alias of rows_with_predictions in this evaluation output.",
        "rows_with_valid_gt": "Rows with canonicalizable ground-truth reactant sets.",
        "rows_with_valid_gt_template": "Rows with at least one valid ground-truth template label.",
        "template_attempts_total": "Total number of template applications attempted across all rows.",
        "template_success_total": "Total number of attempted templates that generated at least one outcome.",
        "top{k}_accuracy": "Exact reactant-set hit rate: fraction of rows where ground-truth reactants appear in top-k ranked reactant-set predictions.",
        "template_top{k}_accuracy": "Template retrieval hit rate: fraction of rows where at least one ground-truth template appears in top-k retrieved templates.",
        "template_applicability_rate@k": "Fraction of top-k templates that can be applied to the product and generate at least one reactant outcome.",
        "mean_unique_reactant_sets_per_product@k": "Average number of unique canonical reactant sets generated from top-k templates per product.",
        "gt_template_yield_count@k": "Average number of top-k templates per row that lead to the ground-truth reactants; deduplication is done within each template's outcomes, not across templates.",
        "gt_template_yield_rate@k": "What fraction of top-k templates lead to GT reactants? Deduplication is done within each template's outcomes, not across templates.",
        "gt_template_yield_coverage@k": "For how many rows does at least one top-k template lead to GT reactants? Deduplication is done within each template's outcomes, not across templates.",
    }

    summary: Dict[str, Any] = {
        "run_info": {
            **arg_snapshot,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "artifact_dir": str(artifact.artifact_dir),
            "stage": artifact.stage,
            "config_snapshot": str(artifact.config_snapshot),
            "code_snapshot": str(artifact.code_snapshot) if artifact.code_snapshot else None,
        },
        "rows_with_predictions": counters.rows_with_predictions,
        "total": counters.rows_with_predictions,
        "rows_with_valid_gt": counters.rows_with_valid_ground_truth,
        "rows_with_valid_gt_template": counters.rows_with_valid_ground_truth_template,
        "template_attempts_total": counters.template_attempts_total,
        "template_success_total": counters.template_success_total,
        **exact_rates,
        **template_retrieval_rates,
        **template_applicability_rates,
        **reactant_set_diversity_rates,
        **gt_template_yield_count_rates,
        **gt_template_yield_rate_rates,
        **gt_template_yield_coverage_rates,
        "metric_definitions": metric_definitions,
    }

    if row_validator_enabled:
        total_seen = sum(filter_reason_counts.values())
        summary["validation_filter"] = {
            "enabled": True,
            "dropped_reason_counts": dict(sorted(filter_reason_counts.items())),
            "dropped_rows": total_seen,
        }

    return summary


def _write_summary(summary: Dict[str, Any], summary_out: Optional[str]) -> None:
    """Write summary JSON to disk and print it to stdout.

    Args:
        summary: Populated summary dict.
        summary_out: File path to write to (None = skip file write).
    """
    if summary_out:
        with open(summary_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    print(json.dumps({"_summary": summary}, indent=2))


def _parse_topk_values(raw: str) -> List[int]:
    """Parse comma-separated top-k values into a sorted integer list."""
    values = {int(t) for t in (tok.strip() for tok in raw.split(",")) if t.isdigit() and int(t) > 0}
    return sorted(values or {1, 3, 5, 10})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Evaluate retrosynthesis predictions from an input JSONL dataset."""
    args = _parse_args()
    _validate_args(args)

    rdkit = RDKitOps.load()
    top_k_values = _parse_topk_values(args.eval_k)

    # --- Artifacts ---
    config = load_config(args.config)
    artifact = _setup_artifacts(
        config,
        args.checkpoint,
        args.artifact_dir,
        args.artifact_name,
        args.save_code_snapshot,
    )
    print(f"[artifact] writing run outputs to {artifact.artifact_dir}")
    print(f"[artifact] saved config to {artifact.config_snapshot}")
    if artifact.code_snapshot:
        print(f"[artifact] saved project snapshot to {artifact.code_snapshot}")
    else:
        print("[artifact] code snapshot disabled")

    # Resolve output file paths relative to artifact_dir when not absolute.
    summary_out: Optional[str] = args.summary_out
    if args.summary and not summary_out:
        summary_out = str(artifact.artifact_dir / "summary.json")
    elif summary_out and not Path(summary_out).is_absolute():
        summary_out = str(artifact.artifact_dir / summary_out)

    predictions_out: Optional[str] = args.predictions_out
    if predictions_out and not Path(predictions_out).is_absolute():
        predictions_out = str(artifact.artifact_dir / predictions_out)
    top_templates_out: Optional[str] = None
    if args.save_top_max_k_templates:
        top_templates_out = str(artifact.artifact_dir / "top_max_k_templates.jsonl")
        print(
            f"[artifact] will save top-max-k templates per product to {top_templates_out} "
            f"(k={max(top_k_values)})"
        )

    eval_cfg = EvalConfig.from_args(args)
    eval_cfg = EvalConfig(
        **{**eval_cfg.__dict__, "summary_out": summary_out, "predictions_out": predictions_out}
    )

    # --- Model ---
    model_device = _resolve_eval_device(args.device)
    model, tokenizer = _load_model(config, args.checkpoint, model_device)
    product_input_builder = get_text_input_builder(config, tokenizer, "product")
    print(f"[inference] model_device={model_device}")

    # --- Templates ---
    templates, template_embeddings_cpu, cache_fp16 = _load_template_embeddings(
        cache_path=args.cache,
        templates_path=args.templates,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        rebuild_cache=args.rebuild_cache,
    )
    faiss_index = _load_faiss_index(args.faiss_index, args.faiss_nprobe, args.faiss_ef_search)

    use_fp16 = bool(cache_fp16 and model_device.type == "cuda")
    template_embeddings_device: Optional[torch.Tensor] = None
    if faiss_index is None:
        if template_embeddings_cpu is None:
            raise RuntimeError("Template embeddings not available. Provide --cache or --faiss_index.")
        dtype = torch.float16 if use_fp16 else torch.float32
        template_embeddings_device = template_embeddings_cpu.to(device=model_device, dtype=dtype)
        print(
            f"[inference] preloaded template matrix "
            f"shape={tuple(template_embeddings_device.shape)} "
            f"dtype={template_embeddings_device.dtype} "
            f"device={template_embeddings_device.device}"
        )
    template_embeddings_cpu = None  # free CPU copy

    retrieval_ctx = RetrievalContext(
        model=model,
        product_input_builder=product_input_builder,
        templates=templates,
        template_embeddings_device=template_embeddings_device,
        faiss_index=faiss_index,
        model_device=model_device,
        use_fp16_similarity=use_fp16,
    )

    # --- Evaluation loop ---
    row_validator = build_row_validator_from_config(config, split="test")
    counters, filter_reason_counts = _run_evaluation_loop(
        eval_jsonl=args.eval_jsonl,
        retrieval_ctx=retrieval_ctx,
        eval_cfg=eval_cfg,
        top_k_values=top_k_values,
        row_validator=row_validator,
        rdkit=rdkit,
        top_templates_out=top_templates_out,
        top_templates_k=max(top_k_values),
    )

    if not eval_cfg.summary:
        return

    # --- Summary ---
    summary = _build_summary(
        counters=counters,
        top_k_values=top_k_values,
        artifact=artifact,
        arg_snapshot=_build_arg_snapshot(args),
        filter_reason_counts=filter_reason_counts,
        row_validator_enabled=row_validator is not None,
    )
    _write_summary(summary, eval_cfg.summary_out)


if __name__ == "__main__":
    main()
