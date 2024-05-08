from uuid import UUID
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import TextSplitter
from chromadb import Collection as ChromaCollection

from uni_ir.store import Document

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

    def search(self, query: str, k: int) -> list[UUID]:
        query_embedding = self.embeddings.embed_query(query)
        result = self.collection.query([query_embedding], n_results=k)

        seen = set()
        docs: list[UUID] = []

        if not result["metadatas"]:
            return []

        for metadata in result["metadatas"][0]:
            parent_idx = metadata["parent_idx"]
            if isinstance(parent_idx, UUID) and parent_idx not in seen:
                seen.add(parent_idx)
                docs.append(parent_idx)

        return docs

    def _index(self, docs: list[Document]) -> None:
        chunks = []
        metadatas = []
        ids = []

        for doc in docs:
            doc_chunks = self.child_splitter.split_text(doc.content)
            chunks.extend(doc_chunks)
            metadatas.extend([{"parent_idx": doc.id}] * len(doc_chunks))
            ids.extend([f"{str(doc.id)}_{i}" for i in range(len(doc_chunks))])

        document_embeddings = self.embeddings.embed_documents(chunks)

        self.collection.add(
            ids=ids,
            embeddings=document_embeddings,  # type: ignore
            documents=chunks,
            metadatas=metadatas,
        )
