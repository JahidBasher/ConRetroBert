from .datatypes import Stage2KLDConfig
from .stage2 import Stage2KLDLightningModule

# Backward-compatibility alias so callers that import Stage2LightningModule from
# this package continue to work without changes.
Stage2LightningModule = Stage2KLDLightningModule

__all__ = [
    "Stage2KLDConfig",
    "Stage2KLDLightningModule",
    "Stage2LightningModule",
]
