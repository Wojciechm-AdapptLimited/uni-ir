from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Generic, TypeVar
from uuid import UUID, uuid4
from pydantic import BaseModel


class Entity(BaseModel):
    id: UUID | None = None


T = TypeVar("T", bound=Entity)
U = TypeVar("U")


class BaseStore(ABC, Generic[T]):
    @abstractmethod
    def __setitem__(self, id: UUID, value: T) -> None:
        pass

    @abstractmethod
    def __getitem__(self, id: UUID) -> T:
        pass

    @abstractmethod
    def __delitem__(self, id: UUID) -> None:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __contains__(self, id: UUID) -> bool:
        pass

    def where(self, predicate: Callable[[T], bool]) -> Iterator[T]:
        for value in self:
            if predicate(value):
                yield value

    def select(self, selector: Callable[[T], U]) -> Iterator[U]:
        for value in self:
            yield selector(value)

    def add(self, value: T) -> T:
        value.id = uuid4()
        self[value.id] = value
        return value

    def extend(self, values: list[T]) -> list[T]:
        return [self.add(value) for value in values]
