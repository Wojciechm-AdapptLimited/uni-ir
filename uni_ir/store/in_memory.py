from uuid import UUID

from .base import BaseStore, T


class InMemoryStore(BaseStore[T]):
    def __init__(self):
        self._store: dict[UUID, T] = {}

    def __setitem__(self, id: UUID, value: T) -> None:
        self._store[id] = value

    def __getitem__(self, id: UUID) -> T:
        return self._store[id]

    def __delitem__(self, id: UUID) -> None:
        del self._store[id]

    def __iter__(self):
        return iter(self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, id: UUID) -> bool:
        return id in self._store
