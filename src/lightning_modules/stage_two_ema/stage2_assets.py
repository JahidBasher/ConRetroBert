# Re-export from canonical location.  All logic lives in stage_two.
from ..stage_two.stage2_assets import Stage2TemplateAssetMixin  # noqa: F401

__all__ = ["Stage2TemplateAssetMixin"]
