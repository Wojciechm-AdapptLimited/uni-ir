from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from .document import Metadata


class Predicate(BaseModel, ABC):
    @abstractmethod
    def __call__(self, m: Metadata) -> bool:
        pass


class ComparisonOperator(Enum):
    EQ = 1
    NE = 2
    GT = 3
    GTE = 4
    LT = 5
    LTE = 6


ATTRIBUTE = TypeVar("ATTRIBUTE", str, int, float)


class ComparisonPredicate(Predicate, Generic[ATTRIBUTE]):
    operator: ComparisonOperator = Field(..., title="Comparator")
    attribute: str = Field(..., title="Attribute to compare")
    value: ATTRIBUTE = Field(..., title="Value to compare")

    def __call__(self, m: Metadata) -> bool:
        assert self.attribute in m.model_fields
        attr_val: ATTRIBUTE = m.__getattribute__(self.attribute)
        assert isinstance(attr_val, type(self.value))

        match self.operator:
            case ComparisonOperator.EQ:
                return attr_val == self.value
            case ComparisonOperator.NE:
                return attr_val != self.value
            case ComparisonOperator.GT:
                return attr_val > self.value
            case ComparisonOperator.GTE:
                return attr_val >= self.value
            case ComparisonOperator.LT:
                return attr_val < self.value
            case ComparisonOperator.LTE:
                return attr_val <= self.value
            case _:
                return False


class LogicalOperator(Enum):
    AND = 1
    OR = 2
    NOT = 3


class LogicalPredicate(Predicate):
    operator: LogicalOperator = Field(..., title="Logical operator")
    statements: list[Predicate] = Field(..., title="List of statements")

    def __call__(self, m: Metadata) -> bool:
        match self.operator:
            case LogicalOperator.AND:
                return all(statement(m) for statement in self.statements)
            case LogicalOperator.OR:
                return any(statement(m) for statement in self.statements)
            case LogicalOperator.NOT:
                return not self.statements[0](m)
            case _:
                return False
