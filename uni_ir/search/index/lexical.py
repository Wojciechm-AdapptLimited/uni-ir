from collections import defaultdict
import numpy as np

from typing import Callable
from uuid import UUID

from uni_ir.store import Document

from .base import BaseIndex


class LexicalIndex(BaseIndex):
    def __init__(
        self,
        tokenizer: Callable[[str], list[str]] = lambda x: x.split(),
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.tokenizer = tokenizer
        self.corpus = {}
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self._index([])

    @classmethod
    def from_docs(
        cls,
        docs: list[Document],
        *,
        tokenizer: Callable[[str], list[str]] = lambda x: x.split(),
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> "LexicalIndex":
        index = cls(tokenizer, k1, b, epsilon)
        index.index(docs)
        return index

    def search(self, query: str, k: int) -> list[UUID]:
        tokens = self.tokenizer(query)
        scores = self._score(tokens)
        return [doc for doc, _ in scores[:k]]

    def _index(self, docs: list[Document]) -> None:
        corpus = {doc.id: self.tokenizer(doc.content) for doc in docs if doc.id}
        self.corpus.update(corpus)

        ids: list[UUID] = []
        dls: list[int] = []
        nd: dict[str, int] = defaultdict(int)
        df: list[dict[str, int]] = []

        for doc in self.corpus:
            ids.append(doc)
            dls.append(len(self.corpus[doc]))
            freq: dict[str, int] = defaultdict(int)

            for term in self.corpus[doc]:
                freq[term] += 1
            df.append(freq)

            for term in freq:
                nd[term] += 1

        self.size = len(ids)
        self.ids = ids
        self.dls = np.array(dls)
        self.avgdl = np.mean(self.dls) if self.size > 0 else 0
        self.df = df
        self.idf = self._calc_idf(nd, self.epsilon)

    def _score(self, query: list[str]) -> list[tuple[UUID, float]]:
        if not self.size:
            return []

        scores = np.zeros(self.size, dtype=np.float64)

        for term in query:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            tf = np.array([doc.get(term, 0) for doc in self.df])
            numerator = idf * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * self.dls / self.avgdl)
            scores += numerator / denominator

        return list(sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True))

    def _calc_idf(self, nd: dict[str, int], epsilon: float) -> dict[str, float]:
        if not self.size:
            return {}

        sum_idf = 0
        idf: dict[str, float] = {}

        for term in nd:
            term_idf = np.log((self.size - nd[term] + 0.5) / (nd[term] + 0.5))
            idf[term] = term_idf
            sum_idf += term_idf

        avg_idf = sum_idf / len(nd)

        for term in idf:
            idf[term] = idf[term] if idf[term] > 0 else avg_idf * epsilon

        return idf
