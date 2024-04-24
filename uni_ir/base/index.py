from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class BaseIndex(ABC):
    docs: list[Document]

    @abstractmethod
    def search(self, query: str, k: int) -> list[Document]:
        pass

    def as_retriever(self, k: int = 5):
        return IndexBackedRetriever(index=self, k=k)


class IndexBackedRetriever(BaseRetriever):
    index: BaseIndex
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.index.search(query, k=self.k)
