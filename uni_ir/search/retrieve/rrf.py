from collections import defaultdict
from langchain_core.documents import Document


def rff(rankings: list[list[Document]], weights: list[float], c: int):
    assert len(rankings) == len(weights)
    assert all(1.0 > weight > 0.0 for weight in weights)
    assert c > 0

    docs_by_content = {}
    rrf_by_content = defaultdict(float)

    for ranking, weight in zip(rankings, weights):
        for i, doc in enumerate(ranking):
            docs_by_content[doc.page_content] = doc
            rrf_by_content[doc.page_content] += weight / (i + c)

    final_ranking = sorted(
        docs_by_content.values(),
        key=lambda doc: rrf_by_content[doc.page_content],
        reverse=True,
    )

    return final_ranking
