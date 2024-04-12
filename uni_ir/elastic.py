from langchain_elasticsearch.vectorstores import BaseRetrievalStrategy
from langchain_elasticsearch._utilities import DistanceStrategy


class HybridRetrievalStrategy(BaseRetrievalStrategy):
    def __init__(self, rrf: dict | bool | None):
        self.rrf = rrf

    def query(
        self,
        query_vector: list[float] | None,
        query: str | None,
        *,
        k: int,
        fetch_k: int,
        vector_query_field: str,
        text_field: str,
        filter: list[dict],
        similarity: DistanceStrategy | None,
    ) -> dict:
        knn = {
            "filter": filter,
            "field": vector_query_field,
            "k": k,
            "num_candidates": fetch_k,
            "query_vector": query_vector,
        }
        body = {
            "knn": knn,
            "query": {
                "bool": {
                    "must": {"match": {text_field: query}},
                    "filter": filter,
                }
            },
        }
        if isinstance(self.rrf, dict):
            body["rank"] = {"rrf": self.rrf}
        elif isinstance(self.rrf, bool) and self.rrf is True:
            body["rank"] = {"rrf": {}}

        return body

    def index(
        self,
        dims_length: int | None,
        vector_query_field: str,
        text_field: str,
        similarity: DistanceStrategy | None,
    ) -> dict:
        if similarity is DistanceStrategy.COSINE:
            similarityAlgo = "cosine"
        elif similarity is DistanceStrategy.EUCLIDEAN_DISTANCE:
            similarityAlgo = "l2_norm"
        elif similarity is DistanceStrategy.DOT_PRODUCT:
            similarityAlgo = "dot_product"
        elif similarity is DistanceStrategy.MAX_INNER_PRODUCT:
            similarityAlgo = "max_inner_product"
        else:
            raise ValueError("Invalid similarity algorithm")

        return {
            "mappings": {
                "dynamic_templates": [
                    {
                        "metadata": {
                            "match_mapping_type": "*",
                            "match": "metadata.*",
                            "unmatch": "metadata.loc",
                            "mapping": {"type": "keyword"},
                        },
                    },
                ],
                "properties": {
                    "metadata": {"type": "object"},
                    vector_query_field: {
                        "type": "dense_vector",
                        "dims": dims_length,
                        "index": True,
                        "similarity": similarityAlgo,
                    },
                    text_field: {"type": "text", "similarity": "bm25"},
                },
            },
            "settings": {
                "similarity": {
                    "bm25": {
                        "type": "BM25",
                        "b": 0.75,
                        "k1": 2.0,
                    },
                },
            },
        }
