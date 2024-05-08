from typing import Callable
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .base import BaseIndex

PREPROCESS_FUNC = Callable[[str], list[str]]


class LexicalIndex(BaseIndex):
    preprocess_func: PREPROCESS_FUNC
    vectorizer: BM25Okapi | None
    preprocessed_docs: list[list[str]]

    def __init__(
        self,
        preprocess_func: PREPROCESS_FUNC = lambda x: x.split(),
        docs: list[Document] | None = None,
        preprocessed_docs: list[list[str]] | None = None,
    ) -> None:
        super().__init__(docs)
        self.preprocess_func = preprocess_func
        self.preprocessed_docs = preprocessed_docs or []

        if preprocessed_docs:
            self.vectorizer = BM25Okapi(preprocessed_docs)

    def search(self, query: str, k: int) -> list[Document]:
        if not self.vectorizer:
            return self.docs

        preprocessed_query = self.preprocess_func(query)
        return self.vectorizer.get_top_n(preprocessed_query, self.docs, k)

    def index(self, docs: list[Document]) -> None:
        preprocessed_docs = [self.preprocess_func(doc.page_content) for doc in docs]
        self.docs.extend(docs)
        self.preprocessed_docs.extend(preprocessed_docs)
        self.vectorizer = BM25Okapi(preprocessed_docs)
