"""Task strategy module for handling different task types."""

from .base import TaskStrategy
from .classification import ClassificationStrategy
from .style_transfer import StyleTransferStrategy
from .reasoning import ReasoningStrategy
from .jailbreaking import JailbreakingStrategy

__all__ = [
    'TaskStrategy',
    'ClassificationStrategy',
    'StyleTransferStrategy',
    'ReasoningStrategy',
    "JailbreakingStrategy"
]
