from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from chromadb import Collection as ChromaCollection


class DenseRetriever(BaseRetriever):
    collection: ChromaCollection
    embeddings: Embeddings
    docs: list[Document]
    k: int

    @classmethod
    def from_docs(
        cls,
        docs: list[Document],
        *,
        collection: ChromaCollection,
        embeddings: Embeddings,
        k: int = 5,
    ) -> "DenseRetriever":
        vector_store = cls(collection=collection, embeddings=embeddings, docs=[], k=k)
        vector_store._add(docs)
        return vector_store

    @classmethod
    def from_existing(
        cls,
        docs: list[Document],
        *,
        collection: ChromaCollection,
        embeddings: Embeddings,
        k: int = 5,
    ) -> "DenseRetriever":
        return cls(collection=collection, embeddings=embeddings, docs=docs, k=k)

    def _add(self, docs: list[Document]) -> None:
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

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
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
