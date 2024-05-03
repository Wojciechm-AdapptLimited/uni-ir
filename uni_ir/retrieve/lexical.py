from typing import Callable
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class LexicalIndexBackedRetriever(BaseRetriever):
    preprocess_func: Callable[[str], list[str]]
    vectorizer: BM25Okapi
    docs: list[Document]
    k: int

    @classmethod
    def from_docs(
        cls,
        docs: list[Document],
        *,
        k: int = 5,
        preprocess_func: Callable[[str], list[str]] = lambda x: x.split()
    ) -> "LexicalIndexBackedRetriever":
        preprocessed_docs = [preprocess_func(doc.page_content) for doc in docs]
        vectorizer = BM25Okapi(preprocessed_docs)
        return cls(
            vectorizer=vectorizer, docs=docs, k=k, preprocess_func=preprocess_func
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        preprocessed_query = self.preprocess_func(query)
        return self.vectorizer.get_top_n(preprocessed_query, self.docs, n=self.k)
