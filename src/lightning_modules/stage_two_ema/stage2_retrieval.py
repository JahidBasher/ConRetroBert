# Re-export from canonical location.  All logic lives in stage_two.
from ..stage_two.stage2_retrieval import Stage2RetrievalMixin  # noqa: F401

__all__ = ["Stage2RetrievalMixin"]
