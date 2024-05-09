from uni_ir.store import Document
from uni_ir.store.filter import Predicate

from .base import BaseRetriever
from .fusion import rff


class WeightedRetriever(BaseRetriever):
    def __init__(self, retriever: BaseRetriever, weight: float = 1.0):
        self.retriever = retriever
        self.weight = weight

    def retrieve(
        self, query: str, predicate: Predicate | None = None
    ) -> list[Document]:
        return self.retriever.retrieve(query, predicate=predicate)


class HybridRetriever(BaseRetriever):
    def __init__(self, retrievers: list[WeightedRetriever], c: int = 60):
        self.retrievers = retrievers
        self.c = c

    def retrieve(
        self, query: str, predicate: Predicate | None = None
    ) -> list[Document]:
        rankings = [
            retriever.retrieve(query, predicate=predicate)
            for retriever in self.retrievers
        ]
        weights = [retriever.weight for retriever in self.retrievers]
        weights = _normalize_weights(weights)

        return rff(rankings, weights, c=self.c)


def _normalize_weights(weights: list[float]) -> list[float]:
    total = sum(weights)
    return [w / total for w in weights]
