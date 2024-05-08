from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BaseIndex(ABC):
    docs: list[Document]

    def __init__(self, docs: list[Document] | None = None) -> None:
        self.docs = docs or []

    @abstractmethod
    def search(self, query: str, k: int) -> list[Document]:
        pass

    @abstractmethod
    def index(self, docs: list[Document]) -> None:
        pass
