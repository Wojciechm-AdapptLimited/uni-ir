from abc import ABC, abstractmethod

from uni_ir.search.index import BaseIndex
from uni_ir.store import Document, BaseStore
from uni_ir.store.filter import Predicate


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self, query: str, predicate: Predicate | None = None
    ) -> list[Document]:
        pass


class IndexBackedRetriever(BaseRetriever):
    def __init__(
        self, index: BaseIndex, store: BaseStore[Document], k: int = 5
    ) -> None:
        self.index = index
        self.store = store
        self.k = k

    def retrieve(
        self, query: str, predicate: Predicate | None = None
    ) -> list[Document]:
        indices = self.index.search(query, self.k, predicate=predicate)
        return [self.store[doc] for doc in indices if doc in self.store]
