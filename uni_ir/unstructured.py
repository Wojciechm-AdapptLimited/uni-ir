from typing import Any, Iterator

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

from unstructured.documents.base import Element
from unstructured.partition.pdf import partition_pdf


class UnstructuredLoader(BaseLoader):
    path: str
    unstructured_kwargs: dict[str, Any]

    def __init__(self, path: str, **unstructured_kwargs: dict[str, Any]):
        self.path = path
        self.unstructured_kwargs = unstructured_kwargs

    def lazy_load(self) -> Iterator[Document]:
        elements = self._get_elements()

        page_text: dict[int, str] = {}
        page_metadata: dict[int, dict[str, Any]] = {}

        for element in elements:
            metadata = self._get_metadata()
            metadata.update(element.metadata.to_dict())

            page_number = metadata.get("page_number", 1)
            text = ""

            if hasattr(element, "category") and element.category == "Table":
                text = element.metadata.text_as_html or ""
            else:
                text = str(element)

            # Check if this page_number already exists in docs_dict
            if page_number not in page_text:
                # If not, create new entry with initial text and metadata
                page_text[page_number] = text + "\n\n"
                page_metadata[page_number] = metadata
            else:
                # If exists, append to text and update the metadata
                page_text[page_number] += text + "\n\n"
                page_metadata[page_number].update(metadata)

        for page_number, text in page_text.items():
            yield Document(
                page_content=text,
                metadata=page_metadata[page_number],
            )

    def _get_elements(self) -> list[Element]:
        elements = partition_pdf(
            filename=self.path, languages=["en"], infer_table_structure=True
        )
        return elements

    def _get_metadata(self) -> dict[str, Any]:
        return {"source": self.path}
