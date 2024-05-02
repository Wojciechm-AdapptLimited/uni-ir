from collections import defaultdict
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, patch_config


class HybridRetriever(BaseRetriever):
    def __init__(self, retrievers: list[BaseRetriever], weights: list[float], c=60):
        assert len(retrievers) == len(weights)
        assert all(weight > 0 for weight in weights)
        assert c > 0

        self.retrievers = retrievers
        self.weights = weights
        self.c = c

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

        return _rff(rankings, self.weights, c=self.c)


def _rff(rankings: list[list[Document]], weights: list[float], c: int):
    docs_by_content = {}
    rrf_by_content = defaultdict(float)

    for ranking, weight in zip(rankings, weights):
        for i, doc in enumerate(ranking):
            docs_by_content[doc.page_content] = doc
            rrf_by_content[doc.page_content] += weight / (i + c)

    final_ranking = sorted(
        docs_by_content.values(),
        key=lambda doc: rrf_by_content[doc.page_content],
        reverse=True,
    )

    return final_ranking
