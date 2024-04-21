import base64
import os
import re
import numpy as np

from io import BytesIO
from typing import IO, Any
from dataclasses import dataclass
from PIL import Image as PILImage
from transformers import Blip2ForConditionalGeneration, Blip2Processor, TensorType
from unstructured.partition.auto import partition, PartitionStrategy
from unstructured.documents.elements import (
    Element,
    Table,
    Image,
)
from unstructured.cleaners.core import clean
from unstructured.partition.text_type import sent_tokenize
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from unstructured.staging.base import elements_from_json, elements_to_json
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Text:
    content: str
    page_number: int


class ImageCaptioner:
    def __init__(self, processor: Blip2Processor, model: Blip2ForConditionalGeneration):
        self.processor = processor
        self.model = model

    def caption(
        self,
        payload: str,
    ) -> str:
        if not payload:
            return ""

        content = PILImage.open(BytesIO(base64.b64decode(payload))).convert("RGB")

        inputs = self.processor(
            content, "an image of", return_tensors=TensorType.PYTORCH
        )

        outputs = self.model.generate(
            inputs.pixel_values, inputs.input_ids, inputs.attention_mask
        )

        return self.processor.decode(outputs[0], skip_special_tokens=True).strip()


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

    def load(self, file: IO[bytes], filename: str | None = None) -> list[Document]:
        cache_path = f"{filename}.elements.json"

        if os.path.exists(cache_path):
            elements = elements_from_json(cache_path)
        else:
            elements = partition(
                file=file,
                strategy=PartitionStrategy.HI_RES,
                languages=self.ocr_languages,
                model_name="yolox",
                extract_image_block_types=[Image.__name__],
                extract_image_block_to_payload=True,
            )
            elements_to_json(elements, filename=cache_path)

        pages = self._parse(elements)
        docs = self._transform(pages)

        if filename:
            for doc in docs:
                doc.metadata.update(_extract(doc.page_content))
                doc.metadata["source"] = filename

        return docs

    def _parse(self, elements: list[Element]) -> list[Text]:
        pages = {}

        for element in elements:
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

            if element.metadata.page_number in pages:
                pages[element.metadata.page_number] += text
            else:
                pages[element.metadata.page_number] = text

        return [
            Text(
                content=page_content,
                page_number=page_number,
            )
            for page_number, page_content in pages.items()
        ]

    def _transform(self, texts: list[Text]) -> list[Document]:
        chunks = _chunk(sorted(texts, key=lambda x: x.page_number))
        embeddings = self.embeddings.embed_documents(
            [chunk.content for chunk in _combine_by_proximity(chunks)]
        )
        distances = _calculate_distances(embeddings)
        combined_chunks = _combine_by_similarity(
            chunks, distances, self.chunking_percentile
        )
        combined_chunks = _combine_by_size(combined_chunks, self.min_chunk_size)

        return [
            Document(
                page_content=chunk.content, metadata={"page_number": chunk.page_number}
            )
            for chunk in combined_chunks
        ]


def _chunk(texts: list[Text]) -> list[Text]:
    chunks = []

    for text in texts:
        sentences = _split(text.content)

        for sentence in sentences:
            chunk = Text(
                content=sentence,
                page_number=text.page_number,
            )
            chunks.append(chunk)

    return chunks


def _combine_by_proximity(chunks: list[Text], buffer_size=1) -> list[Text]:
    combined_chunks = []

    for i, chunk in enumerate(chunks):
        combined_sentence = ""
        page_number = None

        for j in range(i - buffer_size, i):
            if j < 0:
                continue
            page_number = page_number or chunks[j].page_number
            combined_sentence += chunks[j].content

        combined_sentence += chunk.content

        for j in range(i + 1, i + buffer_size + 1):
            if j >= len(chunks):
                continue
            combined_sentence += chunks[j].content

        combined_chunks.append(
            Text(
                content=combined_sentence,
                page_number=page_number or chunk.page_number,
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
            Text(content=combined_sentence, page_number=group[0].page_number)
        )

        start_idx = breakpoint_idx + 1

    if start_idx < len(chunks):
        group = chunks[start_idx:]
        combined_sentence = " ".join([c.content for c in group])
        combined_chunks.append(
            Text(content=combined_sentence, page_number=group[0].page_number)
        )

    return combined_chunks


def _combine_by_size(chunks: list[Text], min_chunk_size: int) -> list[Text]:
    combined_chunks = []

    for i in range(len(chunks) - 1):

        if len(chunks[i].content) >= min_chunk_size:
            combined_chunks.append(chunks[i])
            continue

        chunks[i + 1].content = chunks[i].content + " " + chunks[i + 1].content
        chunks[i + 1].page_number = chunks[i].page_number

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


def _extract(text: str) -> dict[str, Any]:

    return {}
