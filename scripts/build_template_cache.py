import argparse
import json
import os
import tqdm
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.input_processing import collate_feature_dicts, get_text_input_builder
from src.model import build_model_from_config
from src.utils import load_config, load_or_build_tokenizer, load_weights_into_model


def load_templates(path: str) -> List[str]:
    templates = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tmpl = obj.get("template", "")
            if tmpl and tmpl not in seen:
                seen.add(tmpl)
                templates.append(tmpl)
    return templates


class TemplateTextDataset(Dataset):
    def __init__(self, templates: List[str]) -> None:
        self.templates = templates

    def __len__(self) -> int:  # pragma: no cover
        return len(self.templates)

    def __getitem__(self, idx: int) -> str:
        return self.templates[idx]


def _identity_batch_collator(batch):
    return batch


def _to_device(feature_dict, device: torch.device, non_blocking: bool):
    return {
        k: v.to(device=device, non_blocking=non_blocking) if torch.is_tensor(v) else v
        for k, v in feature_dict.items()
    }


def _is_cuda_device(device: torch.device) -> bool:
    return str(device).startswith("cuda")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--templates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", help="Enable DataLoader pin_memory for CUDA")
    parser.add_argument("--faiss_index", default=None, help="Optional FAISS index output path")
    parser.add_argument("--faiss_index_type", default="flat", choices=["flat", "ivf", "hnsw"])
    parser.add_argument("--faiss_nlist", type=int, default=1024, help="IVF: number of clusters")
    parser.add_argument("--faiss_nprobe", type=int, default=16, help="IVF: probes at search time")
    parser.add_argument("--faiss_m", type=int, default=32, help="HNSW: M parameter")
    parser.add_argument("--faiss_ef_construction", type=int, default=200, help="HNSW: efConstruction")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = load_or_build_tokenizer(cfg, allow_build=False)
    template_input_builder = get_text_input_builder(cfg, tokenizer, "template")
    model = build_model_from_config(cfg, tokenizer)
    load_weights_into_model(model, args.checkpoint)

    model.eval()
    device = torch.device(args.device)
    model.to(device)
    templates = load_templates(args.templates)
    if not templates:
        raise RuntimeError("No templates found in JSONL.")

    use_cuda = _is_cuda_device(device)
    non_blocking = use_cuda
    output_dtype = torch.float16 if args.fp16 else torch.float32

    use_pin_memory = bool(args.pin_memory and use_cuda)
    dataset = TemplateTextDataset(templates)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_identity_batch_collator,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    embeddings = None
    next_offset = 0
    for chunk in tqdm.tqdm(data_loader, desc="Encoding templates"):
        features = [template_input_builder(tmpl) for tmpl in chunk]
        tmpl_inputs = collate_feature_dicts(features)
        tmpl_inputs = _to_device(tmpl_inputs, device, non_blocking)
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=use_cuda):
                _, tmpl_cls = model.encode_template(tmpl_inputs)
                tmpl_cls = F.normalize(tmpl_cls, dim=-1)
        tmpl_cls_cpu = tmpl_cls.detach().cpu()
        if embeddings is None:
            embeddings = torch.empty((len(templates), tmpl_cls_cpu.shape[1]), dtype=output_dtype, device="cpu")
        embeddings[next_offset : next_offset + len(chunk)] = tmpl_cls_cpu.to(output_dtype)
        next_offset += len(chunk)

    tmpl_cls = embeddings
    if args.fp16 and tmpl_cls.dtype != torch.float16:
        # Preserve legacy behavior if we ever change output_dtype logic above.
        tmpl_cls = tmpl_cls.half()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"templates": templates, "embeddings": tmpl_cls.cpu()}, args.output)
    print(f"Saved cache with {len(templates)} templates to {args.output}")

    if args.faiss_index:
        try:
            import faiss
        except Exception as exc:
            raise RuntimeError("faiss is not installed; cannot build FAISS index.") from exc
        emb = tmpl_cls.cpu().float().numpy()
        dim = emb.shape[1]
        if args.faiss_index_type == "flat":
            index = faiss.IndexFlatIP(dim)
        elif args.faiss_index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, args.faiss_nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(emb)
            index.nprobe = args.faiss_nprobe
        else:
            index = faiss.IndexHNSWFlat(dim, args.faiss_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = args.faiss_ef_construction
        index.add(emb)
        faiss.write_index(index, args.faiss_index)
        print(f"Saved FAISS index to {args.faiss_index}")


if __name__ == "__main__":
    main()
