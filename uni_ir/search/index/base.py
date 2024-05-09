from abc import ABC, abstractmethod
from uuid import UUID

from uni_ir.store import Document
from uni_ir.store.filter import Predicate


class BaseIndex(ABC):
    @abstractmethod
    def search(
        self, query: str, k: int, predicate: Predicate | None = None
    ) -> list[UUID]:
        pass

    def index(self, docs: list[Document]) -> None:
        assert all(doc.id for doc in docs), "All documents must have IDs"
        self._index(docs)

    @abstractmethod
    def _index(self, docs: list[Document]) -> None:
        pass
