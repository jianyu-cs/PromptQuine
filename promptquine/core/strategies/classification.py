"""Classification task strategy implementation."""

from .base import TaskStrategy
from promptquine.utils.classification import (
    ClassificationPromptCandidate,
    get_classification_field,
    aggregate_classification_results,
    should_be_evaluated_next_round_for_classification
)
from promptquine.examples.classification.fsc_evaluator import (
    PromptedClassificationEvaluator
)


class ClassificationStrategy(TaskStrategy):
    """Strategy for classification tasks."""

    @classmethod
    def create_candidate(cls, prompt, mask, eval_result, tokenizer):
        return ClassificationPromptCandidate.from_evaluation(
            prompt, mask, eval_result, tokenizer
        )
    
    @classmethod
    def should_evaluate_next_round(cls, eval_result, min_reward, reward_driven=None):
        return should_be_evaluated_next_round_for_classification(
            eval_result, min_reward, reward_driven
        )
    
    @classmethod
    def get_field(cls, lst, key, default=None, reward_driven=None):
        return get_classification_field(lst, key, default, reward_driven)

    @classmethod
    def create_evaluator(cls, **kwargs):
        return PromptedClassificationEvaluator(**kwargs)
    
    @classmethod
    def aggregate_results(cls, results):
        return aggregate_classification_results(results)
