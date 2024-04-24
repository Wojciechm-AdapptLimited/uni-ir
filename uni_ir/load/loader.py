import os
import re
import numpy as np
import json

from typing import IO
from dataclasses import dataclass

from unstructured.partition.auto import (
    partition,
    PartitionStrategy,
    detect_filetype,
    FileType,
)
from unstructured.documents.elements import (
    Element,
    Table,
    Image,
)
from unstructured.cleaners.core import clean
from unstructured.partition.text_type import sent_tokenize
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

from uni_ir.load import ImageCaptioner


@dataclass
class Text:
    content: str
    section: str | None


class DocumentLoader:
    def __init__(
        self,
        embeddings: Embeddings,
        image_captioner: ImageCaptioner,
        ocr_languages: list[str] | None = None,
        chunking_percentile=90,
        min_chunk_size=2000,
    ):
        self.embeddings = embeddings
        self.image_captioner = image_captioner
        self.ocr_languages = ocr_languages or ["en"]
        self.chunking_percentile = chunking_percentile
        self.min_chunk_size = min_chunk_size

    def load(self, file: IO[bytes], filename: str) -> list[Document]:
        filetype = detect_filetype(filename)

        if not filetype or filetype in [FileType.UNK, FileType.EMPTY]:
            return []

        cache_path = f"./cache/{filename}.parsed.json"

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                docs = json.load(f)
            return [Document(**doc) for doc in docs]

        elements = partition(
            file=file,
            strategy=PartitionStrategy.HI_RES,
            languages=self.ocr_languages,
            model_name="yolox",
            extract_image_block_types=[Image.__name__],
            extract_image_block_to_payload=True,
            include_page_breaks=True,
        )

        texts = self._parse(elements)
        chunks = self._chunk(texts)

        docs = [
            Document(
                page_content=chunk.content,
                metadata={"source": filename, "section": chunk.section},
            )
            for chunk in chunks
        ]

        with open(cache_path, "w") as f:
            json.dump(
                [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in docs
                ],
                f,
            )

        return docs

    def _parse(self, elements: list[Element]) -> list[Text]:
        sections = {}

        for element in elements:
            section = element.metadata.page_number
            text = ""

            match element:
                case Image():
                    payload = element.metadata.image_base64 or ""
                    text = f"(Image: {self.image_captioner.caption(payload)})"
                case Table() if element.metadata.text_as_html:
                    text = "\n" + element.metadata.text_as_html
                case _:
                    text = element.text

            text = clean(
                text,
                bullets=True,
                dashes=True,
                trailing_punctuation=True,
                extra_whitespace=True,
            )

            if len(text.strip()) == 0:
                continue

            text += ". "

            if section in sections:
                sections[section] += text
            else:
                sections[section] = text

        return [
            Text(
                text,
                section,
            )
            for section, text in sections.items()
        ]

    def _chunk(self, texts: list[Text]) -> list[Text]:
        chunks = _chunk(texts)
        embeddings = self.embeddings.embed_documents(
            [chunk.content for chunk in _combine_by_proximity(chunks)]
        )
        distances = _calculate_distances(embeddings)
        combined_chunks = _combine_by_similarity(
            chunks, distances, self.chunking_percentile
        )
        combined_chunks = _combine_by_size(combined_chunks, self.min_chunk_size)

        return combined_chunks


def _chunk(texts: list[Text]) -> list[Text]:
    chunks = []

    for text in texts:
        sentences = _split(text.content)

        for sentence in sentences:
            chunk = Text(content=sentence, section=text.section)
            chunks.append(chunk)

    return chunks


def _combine_by_proximity(chunks: list[Text], buffer_size=1) -> list[Text]:
    combined_chunks = []

    for i, chunk in enumerate(chunks):
        combined_sentence = ""
        section = None

        for j in range(i - buffer_size, i):
            if j < 0:
                continue
            section = section or chunks[j].section
            combined_sentence += chunks[j].content

        combined_sentence += chunk.content

        for j in range(i + 1, i + buffer_size + 1):
            if j >= len(chunks):
                continue
            combined_sentence += chunks[j].content

        combined_chunks.append(
            Text(
                content=combined_sentence,
                section=section or chunk.section,
            )
        )

    return combined_chunks


def _combine_by_similarity(
    chunks: list[Text], distances: list[float], threshold: int
) -> list[Text]:
    combined_chunks = []
    breakpoint_threshold = np.percentile(distances, threshold)
    breakpoints = [i for i, d in enumerate(distances) if d > breakpoint_threshold]

    if not breakpoints:
        return chunks

    start_idx = 0

    for breakpoint_idx in breakpoints:
        group = chunks[start_idx : breakpoint_idx + 1]
        combined_sentence = " ".join([c.content for c in group])

        combined_chunks.append(
            Text(content=combined_sentence, section=group[0].section)
        )

        start_idx = breakpoint_idx + 1

    if start_idx < len(chunks):
        group = chunks[start_idx:]
        combined_sentence = " ".join([c.content for c in group])
        combined_chunks.append(
            Text(content=combined_sentence, section=group[0].section)
        )

    return combined_chunks


def _combine_by_size(chunks: list[Text], min_chunk_size: int) -> list[Text]:
    combined_chunks = []

    for i in range(len(chunks) - 1):

        if len(chunks[i].content) >= min_chunk_size:
            combined_chunks.append(chunks[i])
            continue

        chunks[i + 1].content = chunks[i].content + " " + chunks[i + 1].content
        chunks[i + 1].section = chunks[i].section

    if len(chunks[-1].content) >= min_chunk_size or len(combined_chunks) == 0:
        combined_chunks.append(chunks[-1])
    else:
        combined_chunks[-1].content += " " + chunks[-1].content

    return combined_chunks


def _calculate_distances(embeddings: list[list[float]]) -> list[float]:
    distances = []

    for i in range(len(embeddings) - 1):
        cur, nxt = embeddings[i], embeddings[i + 1]

        similarity = cosine_similarity([cur], [nxt])[0][0]

        distances.append(1 - similarity)

    return distances


def _split(text: str) -> list[str]:
    tables = re.findall(r"<table.*?>.*?</table>", text)
    non_table_content = re.split(r"<table.*?>.*?</table>", text)
    sentences = []

    for table, non_table in zip(tables, non_table_content):
        for paragraph in non_table.split("\n"):
            sentences.extend(sent_tokenize(paragraph))

        # sentences.extend(sent_tokenize(non_table))
        sentences.append(table)

    if len(tables) < len(non_table_content):
        for sentence in non_table_content[len(tables) :]:
            sentences.extend(sent_tokenize(sentence))
    else:
        for table in tables[len(non_table_content) :]:
            sentences.append(table)

    return sentences
