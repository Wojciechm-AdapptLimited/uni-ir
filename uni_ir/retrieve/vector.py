from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Collection as ChromaCollection

from .base import IndexBackedRetriever


class VectorIndexBackedRetriever(IndexBackedRetriever):
    collection: ChromaCollection
    embeddings: Embeddings
    docs: list[Document]

    def __init__(
        self,
        collection: ChromaCollection,
        embeddings: Embeddings,
        docs: list[Document],
        k: int,
    ):
        self.collection = collection
        self.embeddings = embeddings
        self.docs = []
        self.k = k

        chunks = []
        metadatas = []
        ids = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        new_idx = len(self.docs)

        for doc in docs:
            self.docs.append(doc)
            doc_chunks = splitter.split_text(doc.page_content)
            chunks.extend(doc_chunks)
            metadatas.extend([{"parent_idx": new_idx}] * len(doc_chunks))
            ids.extend([f"{new_idx}_{i}" for i in range(len(doc_chunks))])
            new_idx += 1

        document_embeddings = self.embeddings.embed_documents(chunks)

        self.collection.add(ids=ids, embeddings=document_embeddings, documents=chunks, metadatas=metadatas)  # type: ignore

    @classmethod
    def from_docs(
        cls,
        collection: ChromaCollection,
        docs: list[Document],
        *,
        embeddings: Embeddings,
        k: int,
    ) -> "VectorIndexBackedRetriever":
        pass

    def _search(self, query: str) -> list[Document]:
        query_embedding = self.embeddings.embed_query(query)
        result = self.collection.query([query_embedding], n_results=self.k)

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
