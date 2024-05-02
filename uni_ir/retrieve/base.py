from abc import abstractmethod
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class IndexBackedRetriever(BaseRetriever):
    docs: list[Document]
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._search(query)

    @abstractmethod
    def _search(self, query: str) -> list[Document]:
        pass
