# Re-export from canonical location.  All logic lives in stage_two.
from ..stage_two.stage2_metrics import Stage2MetricsMixin  # noqa: F401

__all__ = ["Stage2MetricsMixin"]
