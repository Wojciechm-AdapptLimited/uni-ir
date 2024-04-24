from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Collection as ChromaCollection

from uni_ir.base import BaseIndex


class VectorIndex(BaseIndex):
    collection: ChromaCollection
    embeddings: Embeddings

    def __init__(
        self,
        collection: ChromaCollection,
        embeddings: Embeddings,
        docs: list[Document] | None = None,
    ):
        self.collection = collection
        self.embeddings = embeddings
        self.docs = docs or []

    def add_docs(self, docs: list[Document]):
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

        embeddings = self.embeddings.embed_documents(chunks)

        self.collection.add(ids, embeddings, chunks, metadatas)  # type: ignore

    def search(self, query: str, k: int) -> list[Document]:
        query_embedding = self.embeddings.embed_query(
            f"Represent this sentence for searching relevant passages: {query}"
        )

        result = self.collection.query([query_embedding], n_results=k)

        seen = set()
        docs = []

        for metadata in result["metadatas"][0]:  # type: ignore
            parent_idx = metadata["parent_idx"]
            if isinstance(parent_idx, int) and parent_idx not in seen:
                seen.add(parent_idx)
                docs.append(self.docs[parent_idx])

        return docs
