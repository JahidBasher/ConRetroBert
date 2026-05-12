from typing import Any, Dict
from .stage_one.stage1 import Stage1LightningModule
from .stage_two.stage2 import Stage2LightningModule
from .stage_two_oneopt.stage2 import Stage2OneOptimizerLightningModule
from .stage_two_kld.stage2 import Stage2KLDLightningModule
from .stage_two_ema.stage2 import Stage2LightningModule as Stage2EMALightningModule


def create_lightning_module(cfg: Dict[str, Any], tokenizer: Any):
    stage = int(cfg["training"]["stage"])
    if stage == 1:
        return Stage1LightningModule(cfg, tokenizer)
    if stage == 2:
        stage2_cfg = cfg["training"].get("stage2", {})
        oneopt_cfg = (
            stage2_cfg.get("one_optimizer_refresh", {})
            if isinstance(stage2_cfg, dict)
            else {}
        )
        if isinstance(oneopt_cfg, dict) and bool(oneopt_cfg.get("enabled", False)):
            return Stage2OneOptimizerLightningModule(cfg, tokenizer)
        kld_cfg = stage2_cfg.get("kld", {}) if isinstance(stage2_cfg, dict) else {}
        if isinstance(kld_cfg, dict) and bool(kld_cfg.get("enabled", False)):
            return Stage2KLDLightningModule(cfg, tokenizer)
        ema_cfg = stage2_cfg.get("ema", {}) if isinstance(stage2_cfg, dict) else {}
        if isinstance(ema_cfg, dict) and bool(ema_cfg.get("enabled", False)):
            return Stage2EMALightningModule(cfg, tokenizer)
        return Stage2LightningModule(cfg, tokenizer)
    raise ValueError(f"Unsupported training stage: {stage}")
