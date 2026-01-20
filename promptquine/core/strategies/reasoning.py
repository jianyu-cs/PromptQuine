"""Classification task strategy implementation."""

from .base import TaskStrategy
from promptquine.utils.reasoning import (
    ReasoningPromptCandidate,
    get_reasoning_field,
    aggregate_reasoning_results,
    should_be_evaluated_next_round_for_reasoning
)
from promptquine.examples.reasoning.reasoning_evaluator import (
    PromptedReasoningEvaluator
)


class ReasoningStrategy(TaskStrategy):
    """Strategy for reasoning tasks."""

    @classmethod
    def create_candidate(cls, prompt, mask, eval_result, tokenizer):
        return ReasoningPromptCandidate.from_evaluation(
            prompt, mask, eval_result, tokenizer
        )
    
    @classmethod
    def should_evaluate_next_round(cls, eval_result, min_reward, reward_driven=None):
        return should_be_evaluated_next_round_for_reasoning(
            eval_result, min_reward
        )
    
    @classmethod
    def get_field(cls, lst, key, default=None, reward_driven=None):
        return get_reasoning_field(lst, key, default, reward_driven)

    @classmethod
    def create_evaluator(cls, **kwargs):
        return PromptedReasoningEvaluator(**kwargs)
    
    @classmethod
    def aggregate_results(cls, results):
        return aggregate_reasoning_results(results)
