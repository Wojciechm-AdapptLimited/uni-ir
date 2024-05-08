from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, patch_config
from pydantic.v1 import Field

from .rrf import rff


class WeightedRetriever(BaseRetriever):
    retriever: BaseRetriever
    weight: float = Field(gt=0, lt=1)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.retriever._get_relevant_documents(query, run_manager=run_manager)


class HybridRetriever(BaseRetriever):
    retrievers: list[WeightedRetriever]
    c: int = Field(gt=0, default=60)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        config: RunnableConfig | None = None,
    ) -> list[Document]:
        rankings = [
            retriever.invoke(
                query,
                patch_config(
                    config, callbacks=run_manager.get_child(tag=f"retriever_{i+1}")
                ),
            )
            for i, retriever in enumerate(self.retrievers)
        ]
        weights = [retriever.weight for retriever in self.retrievers]
        weights = _normalize_weights(weights)

        return rff(rankings, weights, c=self.c)


def _normalize_weights(weights: list[float]) -> list[float]:
    total = sum(weights)
    return [w / total for w in weights]
