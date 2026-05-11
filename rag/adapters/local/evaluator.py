from rag.stages.evaluator import Evaluator
from rag.types import EvaluationResult, Message, SearchResult

__all__ = ["NoOpEvaluator"]


class NoOpEvaluator(Evaluator):
    """Passthrough: returns a zeroed `EvaluationResult`.

    Used in stacks where evaluation is not configured. `context_recall` and
    `answer_correctness` are `None` to honour the contract that
    reference-dependent metrics are set only when a reference answer is
    supplied — which the NoOp ignores.
    """

    async def evaluate(
        self,
        query: str,
        context: list[SearchResult],
        answer: Message,
        reference: str | None = None,
    ) -> EvaluationResult:
        return EvaluationResult(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=None,
            answer_correctness=None,
        )
