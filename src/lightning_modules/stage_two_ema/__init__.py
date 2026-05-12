from .stage2 import EmaEncoderFreezeScheduler, Stage2EmaLightningModule, Stage2TrainableTemplateMixin

# Backward-compatibility alias so callers that import Stage2LightningModule from
# this package continue to work without changes.
Stage2LightningModule = Stage2EmaLightningModule

__all__ = [
    "EmaEncoderFreezeScheduler",
    "Stage2EmaLightningModule",
    "Stage2LightningModule",
    "Stage2TrainableTemplateMixin",
]
