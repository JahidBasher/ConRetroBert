import math
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch

from ..data.input_processing import get_feature_collator, get_text_input_builder
from ..losses import contrastive_loss
from ..model import build_model_from_config


class BaseConRetroLightningModule(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any], tokenizer: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        self.model = build_model_from_config(cfg, tokenizer)

        self.stage = int(cfg["training"]["stage"])
        self.temperature = float(cfg["training"]["temperature"])
        self.mlm_weight = float(cfg["model"]["mlm"].get("weight", 1.0))
        self.mlm_prob = float(cfg["model"]["mlm"].get("probability", 0.15))

        self.template_list: Optional[List[str]] = None
        self.product_input_builder = get_text_input_builder(cfg, tokenizer, "product")
        self.template_input_builder = get_text_input_builder(cfg, tokenizer, "template")
        self.product_input_collator = get_feature_collator(cfg, "product")
        self.template_input_collator = get_feature_collator(cfg, "template")

    def compute_contrastive_loss(self, z_p: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        return contrastive_loss(z_prod=z_p, z_templ=z_t, temperature=self.temperature)

    def _extract_product_inputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "product_inputs" in batch:
            return dict(batch["product_inputs"])
        return {
            "input_ids": batch["product_ids"],
            "attention_mask": batch["product_mask"],
        }

    def _extract_template_inputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "template_inputs" in batch:
            return dict(batch["template_inputs"])
        return {
            "input_ids": batch["template_ids"],
            "attention_mask": batch["template_mask"],
        }

    def _mask_batch(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self.tokenizer, "mask_tokens"):
            raise RuntimeError("Tokenizer/input processor does not implement mask_tokens required for MLM.")
        ids = input_ids.tolist()
        masked = []
        labels = []
        for row in ids:
            out = self.tokenizer.mask_tokens(row, self.mlm_prob)
            masked.append(out["input_ids"])
            labels.append(out["labels"])
        return (
            torch.tensor(masked, dtype=torch.long, device=input_ids.device),
            torch.tensor(labels, dtype=torch.long, device=input_ids.device),
        )

    def _maybe_mask_inputs_for_mlm(
        self,
        inputs: Dict[str, torch.Tensor],
        use_mlm: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        if not use_mlm or "input_ids" not in inputs:
            return inputs, None
        masked_ids, labels = self._mask_batch(inputs["input_ids"])
        out = dict(inputs)
        out["input_ids"] = masked_ids
        return out, labels

    def _prepare_pair_inputs(
        self,
        batch: Dict[str, Any],
        use_mlm: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        product_inputs = self._extract_product_inputs(batch)
        template_inputs = self._extract_template_inputs(batch)
        product_inputs, prod_labels = self._maybe_mask_inputs_for_mlm(product_inputs, use_mlm=use_mlm)
        template_inputs, templ_labels = self._maybe_mask_inputs_for_mlm(template_inputs, use_mlm=use_mlm)
        return product_inputs, template_inputs, prod_labels, templ_labels

    def configure_optimizers(self):
        opt_cfg = self.cfg["optimizer"]
        params = [p for p in self.model.parameters()]
        if not params:
            raise RuntimeError("No parameters found.")

        optimizer = torch.optim.AdamW(
            params,
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )

        sched_cfg = self.cfg.get("scheduler", {})
        if not sched_cfg.get("enabled", False):
            return optimizer

        name = sched_cfg.get("name", "cosine")
        if name == "warmup_cosine":
            warmup_steps = int(sched_cfg.get("warmup_steps", 1000))
            total_steps_cfg = sched_cfg.get("total_steps")
            if total_steps_cfg is None:
                total_steps = int(self.trainer.estimated_stepping_batches)
            else:
                total_steps = int(total_steps_cfg)
            min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.01))

            def _lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
                return max(min_lr_ratio, cosine)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
            interval = "step"
        elif name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_cfg.get("t_max", self.cfg["training"]["epochs"]),
                eta_min=sched_cfg.get("eta_min", 1.0e-6),
            )
            interval = "epoch"
        elif name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_cfg.get("step_size", 1),
                gamma=sched_cfg.get("gamma", 0.5),
            )
            interval = "epoch"
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=sched_cfg.get("gamma", 0.95),
            )
            interval = "epoch"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
            },
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"], strict=False)
