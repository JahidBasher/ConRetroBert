"""Factory function for creating the appropriate data module for a training stage."""

from typing import Any, Dict, Optional

from .base import BaseReactionDataModule
from .datatypes import ValidatorFactory
from .stage1 import Stage1ReactionDataModule
from .stage2 import Stage2ReactionDataModule


def create_data_module(
    cfg: Dict[str, Any],
    tokenizer: Any,
    validator_factory: Optional[ValidatorFactory] = None,
) -> BaseReactionDataModule:
    """Instantiate the data module for the configured training stage.

    Routes to :class:`~data.stage1.Stage1ReactionDataModule` or
    :class:`~data.stage2.Stage2ReactionDataModule` based on
    ``cfg["training"]["stage"]``.  The optional *validator_factory* is
    passed through for dependency injection in tests or custom pipelines.

    Args:
        cfg: Full experiment config dict with ``cfg["training"]["stage"]`` set.
        tokenizer: CharTokenizer or compatible encoder.
        validator_factory: Optional injected :data:`ValidatorFactory`; passed
            through to the data module unchanged.

    Returns:
        A configured :class:`Stage1ReactionDataModule` or
        :class:`Stage2ReactionDataModule`.

    Raises:
        ValueError: If ``cfg["training"]["stage"]`` is not 1 or 2.
    """
    stage = int(cfg["training"]["stage"])
    if stage == 1:
        return Stage1ReactionDataModule(cfg, tokenizer, validator_factory)
    if stage == 2:
        return Stage2ReactionDataModule(cfg, tokenizer, validator_factory)
    raise ValueError(f"Unsupported training stage: {stage}")
