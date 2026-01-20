"""Jailbreaking task strategy implementation."""

from .base import TaskStrategy
from promptquine.utils.jailbreaking import (
    JailbreakingPromptCandidate,
    get_jailbreaking_field,
    aggregate_jailbreaking_results,
    should_be_evaluated_next_round_for_jailbreaking
)
from promptquine.examples.jailbreaking.jailbreaking_evaluator import (
    PromptedJailbreakingEvaluator
)


class JailbreakingStrategy(TaskStrategy):
    """Strategy for jailbreaking tasks."""

    @classmethod
    def create_candidate(cls, prompt, mask, eval_result, tokenizer):
        return JailbreakingPromptCandidate.from_evaluation(
            prompt, mask, eval_result, tokenizer
        )
    
    @classmethod
    def should_evaluate_next_round(cls, eval_result, min_reward, reward_driven=None):
        return should_be_evaluated_next_round_for_jailbreaking(
            eval_result, min_reward
        )
    
    @classmethod
    def get_field(cls, lst, key, default=None, reward_driven=None):
        return get_jailbreaking_field(lst, key, default, reward_driven)

    @classmethod
    def create_evaluator(cls, **kwargs):
        return PromptedJailbreakingEvaluator(**kwargs)
    
    @classmethod
    def aggregate_results(cls, results):
        return aggregate_jailbreaking_results(results)

