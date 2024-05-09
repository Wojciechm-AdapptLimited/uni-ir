from typing import Any
from uuid import UUID
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import TextSplitter
from chromadb import Collection as ChromaCollection

from uni_ir.store import Document
from uni_ir.store.filter import (
    ComparisonOperator,
    ComparisonPredicate,
    LogicalOperator,
    LogicalPredicate,
    Predicate,
)

from .base import BaseIndex


class DenseIndex(BaseIndex):
    collection: ChromaCollection
    embeddings: Embeddings
    child_splitter: TextSplitter

    def __init__(
        self,
        collection: ChromaCollection,
        embeddings: Embeddings,
        child_splitter: TextSplitter,
    ) -> None:
        self.collection = collection
        self.embeddings = embeddings
        self.child_splitter = child_splitter

    @classmethod
    def from_docs(
        cls,
        docs: list[Document],
        collection: ChromaCollection,
        embeddings: Embeddings,
        child_splitter: TextSplitter,
    ) -> "DenseIndex":
        index = cls(collection, embeddings, child_splitter)
        index.index(docs)
        return index

    def search(
        self, query: str, k: int, predicate: Predicate | None = None
    ) -> list[UUID]:
        query_embedding = self.embeddings.embed_query(query)
        where = _translate_predicate(predicate) if predicate else None
        result = self.collection.query([query_embedding], n_results=k, where=where)

        seen = set()
        docs: list[UUID] = []

        if not result["metadatas"]:
            return []

        for metadata in result["metadatas"][0]:
            parent_idx = metadata["parent_idx"]
            if isinstance(parent_idx, str) and parent_idx not in seen:
                seen.add(parent_idx)
                docs.append(UUID(parent_idx))

        return docs

    def _index(self, docs: list[Document]) -> None:
        chunks = []
        metadatas = []
        ids = []

        for doc in docs:
            doc_chunks = self.child_splitter.split_text(doc.content)
            doc_metadata = doc.metadata.model_dump()
            doc_metadata["parent_idx"] = str(doc.id)

            for mtd in doc_metadata:
                if doc_metadata[mtd] is None:
                    doc_metadata[mtd] = ""

            chunks.extend(doc_chunks)
            metadatas.extend([doc_metadata.copy()] * len(doc_chunks))
            ids.extend([f"{str(doc.id)}_{i}" for i in range(len(doc_chunks))])

        document_embeddings = self.embeddings.embed_documents(chunks)

        self.collection.add(
            ids=ids,
            embeddings=document_embeddings,  # type: ignore
            documents=chunks,
            metadatas=metadatas,
        )


OPERATORS_TO_CHROMA = {
    ComparisonOperator.EQ: "$eq",
    ComparisonOperator.NE: "$ne",
    ComparisonOperator.LT: "$lt",
    ComparisonOperator.LTE: "$lte",
    ComparisonOperator.GT: "$gt",
    ComparisonOperator.GTE: "$ge",
    LogicalOperator.AND: "$and",
    LogicalOperator.OR: "$or",
    LogicalOperator.NOT: "$not",
}


def _translate_predicate(p: Predicate) -> dict[str, Any]:
    if isinstance(p, ComparisonPredicate):
        operator = OPERATORS_TO_CHROMA[p.operator]
        return {p.attribute: {operator: p.value}}
    elif isinstance(p, LogicalPredicate):
        if p.operator == LogicalOperator.NOT:
            return {
                OPERATORS_TO_CHROMA[p.operator]: [_translate_predicate(p.statements[0])]
            }
        if (
            p.operator in (LogicalOperator.AND, LogicalOperator.OR)
            and len(p.statements) == 1
        ):
            return _translate_predicate(p.statements[0])

        operator = OPERATORS_TO_CHROMA[p.operator]
        return {
            operator: [_translate_predicate(statement) for statement in p.statements]
        }
    return {}
