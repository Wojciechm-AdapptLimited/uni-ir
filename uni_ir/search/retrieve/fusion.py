from collections import defaultdict
from uuid import UUID

from uni_ir.store import Document


def rff(rankings: list[list[Document]], weights: list[float], c: int) -> list[Document]:
    assert len(rankings) == len(weights)
    assert all(1.0 > weight > 0.0 for weight in weights)
    assert c > 0

    rrf: dict[UUID, float] = defaultdict(float)
    docs: dict[UUID, Document] = {}

    for ranking, weight in zip(rankings, weights):
        for i, doc in enumerate(ranking):
            if doc.id is None:
                continue
            docs[doc.id] = doc
            rrf[doc.id] += weight / (i + c)

    final_ranking = sorted(docs, key=lambda doc: rrf[doc], reverse=True)

    return [docs[doc] for doc in final_ranking]
