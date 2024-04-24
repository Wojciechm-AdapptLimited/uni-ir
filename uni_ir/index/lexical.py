import json
import os

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from uni_ir.base import BaseIndex


def preprocess_text(text: str) -> list[str]:
    return text.split()


class LexicalIndex(BaseIndex):
    vectorizer: BM25Okapi

    def __init__(self, docs: list[Document]):
        self.docs = docs

        cache_path = "./cache/processed_docs.json"

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                preprocessed_docs = json.load(f)
        else:
            preprocessed_docs = [preprocess_text(doc.page_content) for doc in docs]
            with open(cache_path, "w") as f:
                json.dump(preprocessed_docs, f)

        self.vectorizer = BM25Okapi(preprocessed_docs)

    def search(self, query: str, k: int) -> list[Document]:
        preprocessed_query = preprocess_text(query)
        return self.vectorizer.get_top_n(preprocessed_query, self.docs, n=k)
