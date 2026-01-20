"""
Task strategy base classes.

This module defines the abstract interface for different task types
(classification, style transfer, reasoning, etc.).

We use it in pruner designs.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional


class TaskStrategy(ABC):
    """
    Abstract base class for task-specific strategies.
    
    Each task type (classification, style transfer, etc.) should implement
    this interface to provide task-specific behavior for candidate creation,
    evaluation, and field extraction.
    """
    
    @classmethod
    @abstractmethod
    def create_candidate(cls, prompt: str, mask: list, 
                        eval_result: Any, tokenizer) -> Any:
        """Create a prompt candidate from evaluation results."""
        pass
    
    @classmethod
    @abstractmethod
    def should_evaluate_next_round(cls, eval_result: Any, 
                                   min_reward: float, 
                                   reward_driven: Optional[bool] = None) -> bool:
        """Determine if a candidate should proceed to next evaluation round."""
        pass
    
    @classmethod
    @abstractmethod
    def get_field(cls, lst: list, key: str, 
                 default=None, reward_driven: Optional[bool] = None) -> Any:
        """Extract a field from candidate's list representation."""
        pass
    
    @classmethod
    @abstractmethod
    def create_evaluator(cls, **kwargs):
        """Create task-specific evaluator."""
        pass

    @classmethod
    @abstractmethod
    def aggregate_results(cls, eval_results):
        """Aggregate evaluation results."""
        pass
    

