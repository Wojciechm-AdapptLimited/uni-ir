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

from uni_ir.store import Document, Metadata

from .captioner import ImageCaptioner
from .chunker import SemanticChunker


class DocumentLoader:
    def __init__(
        self,
        captioner: ImageCaptioner,
        chunker: SemanticChunker,
        ocr_languages: list[str] | None = None,
    ):
        self.captioner = captioner
        self.chunker = chunker
        self.ocr_languages = ocr_languages or ["en"]

    def load(self, file: IO[bytes], uri: str, mimetype: str) -> list[Document]:
        elements = partition(
            file=file,
            strategy=PartitionStrategy.HI_RES,
            languages=self.ocr_languages,
            model_name="yolox",
            extract_image_block_types=[Image.__name__],
            extract_image_block_to_payload=True,
            include_page_breaks=True,
        )

        docs = self._parse(elements, uri, mimetype)
        docs = self.chunker.chunk(docs)

        return docs

    def _parse(
        self, elements: list[Element], uri: str, mimetype: str
    ) -> list[Document]:
        documents = []

        for element in elements:
            section = str(element.metadata.page_number) or element.metadata.section
            text = ""

            match element:
                case Image():
                    payload = element.metadata.image_base64 or ""
                    text = f"(Image: {self.captioner.caption(payload)})"
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

            documents.append(
                Document(
                    content=text,
                    metadata=Metadata(mimetype=mimetype, uri=uri, section=section),
                )
            )

        return documents
