import os
from uuid import UUID
import chromadb
import json

from dotenv import load_dotenv

# from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from unstructured.partition.auto import detect_filetype, FileType, FILETYPE_TO_MIMETYPE

from uni_ir.store import Document, BaseStore, InMemoryStore
from uni_ir.store.filter import (
    ComparisonPredicate,
    ComparisonOperator,
    LogicalOperator,
    LogicalPredicate,
)
from uni_ir.load import ImageCaptioner, DocumentLoader, SemanticChunker
from uni_ir.search.analyze import QueryAnalyzer
from uni_ir.search.index import DenseIndex, LexicalIndex
from uni_ir.search.retrieve import (
    BaseRetriever,
    HybridRetriever,
    IndexBackedRetriever,
    WeightedRetriever,
)

load_dotenv()

functions_model = OllamaFunctions(model="phi3")  # type: ignore
# embeddings = OllamaEmbeddings(
#     model=os.getenv("EMBEDDING_MODEL") or "snowflake-arctic-embed",
#     base_url=os.getenv("EMBEDDING_URL") or "http://localhost:11434",
#     # base_url="http://ai-box:11434",
# )
embeddings = OpenAIEmbeddings()

chroma_client = chromadb.PersistentClient("./cache/chroma")
analyzer = QueryAnalyzer.from_llm(functions_model)


def pretty_print_docs(docs: list[Document]):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"\nDocument {i+1} | {d.metadata.uri} | {d.metadata.mimetype} | {d.metadata.section}:\n\n{d.content}"
                for i, d in enumerate(docs)
            ]
        )
    )


def load() -> BaseStore[Document]:
    print("Loading documents...")

    store_cache_path = "./cache/docs.json"

    store = InMemoryStore[Document]()

    if os.path.exists(store_cache_path):
        print("Loading store from cache...")
        with open(store_cache_path, "r") as f:
            loaded_store: dict[str, str] = json.load(f)
            for doc_id, doc in loaded_store.items():
                store[UUID(doc_id)] = Document.model_validate_json(doc)
        return store

    captioning_processor = Blip2Processor.from_pretrained(
        os.getenv("CAPTIONING_MODEL") or "Salesforce/blip2-opt-2.7b"
    )
    captioning_model = Blip2ForConditionalGeneration.from_pretrained(
        os.getenv("CAPTIONING_MODEL") or "Salesforce/blip2-opt-2.7b"
    )

    document_loader = DocumentLoader(
        captioner=ImageCaptioner(
            processor=captioning_processor, model=captioning_model  # type: ignore
        ),
        chunker=SemanticChunker(embeddings=embeddings, min_chunk_size=2000),
        ocr_languages=["en"],
    )
    docs: list[Document] = []

    for filename in os.listdir("./docs"):
        filetype = detect_filetype(filename)

        if not filetype or filetype in [FileType.UNK, FileType.EMPTY]:
            continue

        cache_path = f"./cache/{filename}.parsed.json"

        print(f"\nLoading {filename}...")

        if os.path.exists(cache_path):
            print("Loading from cache...")
            with open(cache_path, "r") as f:
                loaded_docs = [Document.model_validate(doc) for doc in json.load(f)]
        else:
            print("Parsing...")
            with open(f"./docs/{filename}", "rb") as file:
                loaded_docs = document_loader.load(
                    file,
                    filename,
                    FILETYPE_TO_MIMETYPE.get(filetype, "application/pdf"),
                )
            with open(cache_path, "w") as f:
                json.dump([doc.model_dump() for doc in loaded_docs], f)

        print(f"Loaded {len(loaded_docs)} documents.")

        docs.extend(loaded_docs)

    store.extend(docs)

    with open(store_cache_path, "w") as f:
        json.dump({str(doc.id): doc.model_dump_json() for doc in store}, f)

    print("Loaded documents.", end="\n\n")

    return store


def create_indices(docs: list[Document]) -> tuple[LexicalIndex, DenseIndex]:
    print("\nIndexing documents...")

    lexical_index = LexicalIndex.from_docs(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    collections = [collection.name for collection in chroma_client.list_collections()]

    if "uni_ir" in collections:
        chroma_collection = chroma_client.get_collection("uni_ir")
        dense_index = DenseIndex(chroma_collection, embeddings, splitter)
    else:
        chroma_collection = chroma_client.create_collection(
            name="uni_ir", metadata={"hnsw:space": "cosine"}  # l2 is the default
        )
        dense_index = DenseIndex.from_docs(
            docs, chroma_collection, embeddings, splitter
        )

    print("Indexed documents.", end="\n\n")

    return lexical_index, dense_index


def create_retrieval_pipeline(
    store: BaseStore[Document],
    lexical_index: LexicalIndex,
    dense_index: DenseIndex,
    k: int,
) -> BaseRetriever:
    lexical_retriever = IndexBackedRetriever(lexical_index, store, k=k)
    dense_retriever = IndexBackedRetriever(dense_index, store, k=k)
    hybrid_retriever = HybridRetriever(
        [
            WeightedRetriever(lexical_retriever, 0.4),
            WeightedRetriever(dense_retriever, 0.6),
        ]
    )
    return hybrid_retriever


def search(retriever: BaseRetriever, query: str) -> list[Document]:
    print("\n\n")
    print("-" * 80)
    print("Query:", query)
    print("-" * 80)

    print("\nSearching...", end="\n\n")
    return retriever.retrieve(query)


def main(query: str, k: int):
    store = load()
    lexical_index, dense_index = create_indices([doc for doc in store])
    retriever = create_retrieval_pipeline(store, lexical_index, dense_index, k)

    subqueries = analyzer.analyze(query)

    print("Original query:", query)
    print("Subqueries:", subqueries)

    common_sources = None

    for subquery in subqueries:
        result = retriever.retrieve(subquery)
        sources = {doc.metadata.uri for doc in result}

        if common_sources is None:
            common_sources = sources
        else:
            common_sources &= sources

    if not common_sources:
        print("No common sources found.")
        return

    print(common_sources)

    predicate = LogicalPredicate(
        operator=LogicalOperator.OR,
        statements=[
            ComparisonPredicate(
                operator=ComparisonOperator.EQ, attribute="uri", value=source
            )
            for source in common_sources
        ],
    )

    result = retriever.retrieve(query, predicate=predicate)

    pretty_print_docs(result)


if __name__ == "__main__":
    main("what is promethee I?", k=5)
