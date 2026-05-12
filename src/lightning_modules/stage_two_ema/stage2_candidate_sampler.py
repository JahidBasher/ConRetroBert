# Re-export from canonical location.  All logic lives in stage_two.
from ..stage_two.stage2_candidate_sampler import Stage2CandidateSampler  # noqa: F401

__all__ = ["Stage2CandidateSampler"]
