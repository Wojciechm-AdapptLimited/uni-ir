from typing import IO

from unstructured.partition.auto import (
    partition,
    PartitionStrategy,
)
from unstructured.documents.elements import (
    Element,
    Table,
    Image,
)
from unstructured.cleaners.core import clean
from langchain_core.documents import Document
from pydantic import BaseModel

from .captioner import ImageCaptioner
from .chunker import SemanticChunker


class DocumentLoader(BaseModel):
    image_captioner: ImageCaptioner
    chunker: SemanticChunker
    ocr_languages: list[str]

    def load(self, file: IO[bytes], filename: str) -> list[Document]:
        elements = partition(
            file=file,
            strategy=PartitionStrategy.HI_RES,
            languages=self.ocr_languages,
            model_name="yolox",
            extract_image_block_types=[Image.__name__],
            extract_image_block_to_payload=True,
            include_page_breaks=True,
        )

        docs = self._parse(elements)
        docs = self.chunker.chunk(docs)

        for doc in docs:
            doc.metadata["source"] = filename

        return docs

    def _parse(self, elements: list[Element]) -> list[Document]:
        documents = []

        for element in elements:
            section = element.metadata.page_number or element.metadata.section or "0"
            text = ""

            match element:
                case Image():
                    payload = element.metadata.image_base64 or ""
                    text = f"(Image: {self.image_captioner.caption(payload)})"
                case Table() if element.metadata.text_as_html:
                    text = element.metadata.text_as_html
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

            documents.append(Document(text, metadata={"section": str(section)}))

        return documents
