from langchain_core.documents import Document
from langchain_community.cross_encoders import BaseCrossEncoder
from pydantic import BaseModel


class Reranker(BaseModel):
    """Document compressor that uses CrossEncoder for reranking."""

    model: BaseCrossEncoder
    """CrossEncoder model to use for scoring similarity
      between the query and documents."""
    threshold: float = 0.5
    """Threshold for the similarity score."""

    def rerank(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:
        """
        Rerank documents using CrossEncoder.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.

        Returns:
            A sequence of compressed documents.
        """
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        result = list(zip(documents, scores))

        return [doc for doc, score in result if score >= self.threshold]
