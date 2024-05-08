from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic.v1 import Field

from uni_ir.search.index.base import BaseIndex


class IndexBackedRetriever(BaseRetriever):
    index: BaseIndex
    k: int = Field(gt=0, default=10)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.index.search(query, self.k)
