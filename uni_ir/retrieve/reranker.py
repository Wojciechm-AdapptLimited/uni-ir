from langchain_core.documents import Document
from langchain_community.cross_encoders import BaseCrossEncoder


class Reranker:
    """Document compressor that uses CrossEncoder for reranking."""

    model: BaseCrossEncoder
    """CrossEncoder model to use for scoring similarity
      between the query and documents."""
    threshold: float
    """Threshold for the similarity score."""

    def __init__(self, model: BaseCrossEncoder, threshold: float):
        self.model = model
        self.threshold = threshold

    def rerank(
        self,
        documents: list[Document],
        query: str,
    ) -> list[Document]:
        """
        Rerank documents using CrossEncoder.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        result = list(zip(documents, scores))

        return [doc for doc, score in result if score >= self.threshold]
