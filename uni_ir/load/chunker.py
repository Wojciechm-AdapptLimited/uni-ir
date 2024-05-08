import re
import numpy as np

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sklearn.metrics.pairwise import cosine_similarity
from unstructured.partition.text_type import sent_tokenize
from pydantic import BaseModel


class SemanticChunker(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    embeddings: Embeddings
    combine_buffer_size: int = 1
    chunking_percentile: int = 90
    min_chunk_size: int = 2000

    def chunk(self, documents: list[Document]) -> list[Document]:
        chunks = [chunk for document in documents for chunk in _split(document)]
        embeddings = self.embeddings.embed_documents(
            [
                chunk.page_content
                for chunk in _combine_by_proximity(chunks, self.combine_buffer_size)
            ]
        )
        distances = _calculate_distances(embeddings)
        chunks = _combine_by_similarity(chunks, distances, self.chunking_percentile)
        chunks = _combine_by_size(chunks, self.min_chunk_size)

        return chunks


def _split(document: Document) -> list[Document]:
    tables = re.findall(r"<table.*?>.*?</table>", document.page_content)
    non_table_content = re.split(r"<table.*?>.*?</table>", document.page_content)
    sentences = []

    for table, non_table in zip(tables, non_table_content):
        for paragraph in non_table.split("\n"):
            sentences.extend(sent_tokenize(paragraph))

        sentences.append(table)

    if len(tables) < len(non_table_content):
        for sentence in non_table_content[len(tables) :]:
            sentences.extend(sent_tokenize(sentence))
    else:
        for table in tables[len(non_table_content) :]:
            sentences.append(table)

    return [
        Document(page_content=sentence, metadata=document.metadata)
        for sentence in sentences
    ]


def _combine_by_proximity(chunks: list[Document], buffer_size: int) -> list[Document]:
    combined_chunks = []

    for i, chunk in enumerate(chunks):
        combined_sentence = ""
        section = None

        for j in range(i - buffer_size, i):
            if j < 0:
                continue
            section = section or chunks[j].metadata["section"]
            combined_sentence += chunks[j].page_content

        combined_sentence += chunk.page_content

        for j in range(i + 1, i + buffer_size + 1):
            if j >= len(chunks):
                continue
            combined_sentence += chunks[j].page_content

        section = section or chunk.metadata["section"]
        combined_chunks.append(
            Document(
                page_content=combined_sentence,
                metadata=chunk.metadata | {"section": section},
            )
        )

    return combined_chunks


def _combine_by_similarity(
    chunks: list[Document], distances: list[float], percentile: int
) -> list[Document]:
    def join_chunks(chunks: list[Document]) -> Document:
        page_content = " ".join([c.page_content for c in chunks])
        metadata = chunks[0].metadata | {"section": chunks[0].metadata["section"]}
        return Document(page_content, metadata=metadata)

    combined_chunks = []
    breakpoint_threshold = np.percentile(distances, percentile)
    breakpoints = [i for i, d in enumerate(distances) if d > breakpoint_threshold]

    if not breakpoints:
        return chunks

    start_idx = 0

    for breakpoint_idx in breakpoints:
        group = chunks[start_idx : breakpoint_idx + 1]
        combined_chunks.append(join_chunks(group))
        start_idx = breakpoint_idx + 1

    if start_idx < len(chunks):
        group = chunks[start_idx:]
        combined_chunks.append(join_chunks(group))

    return combined_chunks


def _combine_by_size(chunks: list[Document], size: int) -> list[Document]:
    combined_chunks: list[Document] = []

    for i in range(len(chunks) - 1):

        if len(chunks[i].page_content) >= size:
            combined_chunks.append(chunks[i])
            continue

        chunks[i + 1].page_content = (
            chunks[i].page_content + " " + chunks[i + 1].page_content
        )
        chunks[i + 1].metadata["section"] = chunks[i].metadata["section"]

    if len(chunks[-1].page_content) >= size or len(combined_chunks) == 0:
        combined_chunks.append(chunks[-1])
    else:
        combined_chunks[-1].page_content += " " + chunks[-1].page_content

    return combined_chunks


def _calculate_distances(embeddings: list[list[float]]) -> list[float]:
    distances = []

    for i in range(len(embeddings) - 1):
        cur, nxt = embeddings[i], embeddings[i + 1]

        similarity = cosine_similarity([cur], [nxt])[0][0]

        distances.append(1 - similarity)

    return distances
