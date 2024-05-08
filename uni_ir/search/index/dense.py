from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import TextSplitter
from chromadb import Collection as ChromaCollection

from .base import BaseIndex


class DenseIndex(BaseIndex):
    collection: ChromaCollection
    embeddings: Embeddings
    child_splitter: TextSplitter
    docs: list[Document]

    def __init__(
        self,
        collection: ChromaCollection,
        embeddings: Embeddings,
        child_splitter: TextSplitter,
        docs: list[Document] | None = None,
    ) -> None:
        super().__init__(docs)
        self.collection = collection
        self.embeddings = embeddings
        self.child_splitter = child_splitter

    def search(self, query: str, k: int) -> list[Document]:
        query_embedding = self.embeddings.embed_query(query)
        result = self.collection.query([query_embedding], n_results=k)
        # docs = [self.docs[int(id)] for id in result["ids"][0]]

        seen = set()
        docs = []

        if not result["metadatas"]:
            return []

        for metadata in result["metadatas"][0]:
            parent_idx = metadata["parent_idx"]
            if isinstance(parent_idx, int) and parent_idx not in seen:
                seen.add(parent_idx)
                docs.append(self.docs[parent_idx])

        return docs

    def index(self, docs: list[Document]) -> None:
        chunks = []
        metadatas = []
        ids = []
        new_idx = len(self.docs)

        # for doc in docs:
        #     self.docs.append(doc)
        #     chunks.append(doc.page_content)
        #     ids.append(str(new_idx))
        #     new_idx += 1

        for doc in docs:
            self.docs.append(doc)
            doc_chunks = self.child_splitter.split_text(doc.page_content)
            chunks.extend(doc_chunks)
            metadatas.extend([{"parent_idx": new_idx}] * len(doc_chunks))
            ids.extend([f"{new_idx}_{i}" for i in range(len(doc_chunks))])
            new_idx += 1

        document_embeddings = self.embeddings.embed_documents(chunks)

        self.collection.add(
            ids=ids,
            embeddings=document_embeddings,  # type: ignore
            documents=chunks,
            metadatas=metadatas,
        )
