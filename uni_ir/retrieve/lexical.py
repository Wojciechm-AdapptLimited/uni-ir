from typing import Callable
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from .base import IndexBackedRetriever


class LexicalIndexBackedRetriever(IndexBackedRetriever):
    preprocess_func: Callable[[str], list[str]]
    vectorizer: BM25Okapi

    @classmethod
    def from_docs(
        cls,
        docs: list[Document],
        *,
        k: int,
        preprocess_func: Callable[[str], list[str]]
    ) -> "LexicalIndexBackedRetriever":
        preprocessed_docs = [preprocess_func(doc.page_content) for doc in docs]
        vectorizer = BM25Okapi(preprocessed_docs)
        return cls(
            vectorizer=vectorizer, docs=docs, k=k, preprocess_func=preprocess_func
        )

    def _search(self, query: str) -> list[Document]:
        preprocessed_query = self.preprocess_func(query)
        return self.vectorizer.get_top_n(preprocessed_query, self.docs, n=self.k)
