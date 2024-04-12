import os
from dotenv import load_dotenv

from elasticsearch import Elasticsearch
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_experimental.text_splitter import SemanticChunker
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from uni_ir.elastic import HybridRetrievalStrategy
from uni_ir.unstructured import UnstructuredLoader
from uni_ir.utils.formatting import format_docs, format_query
from uni_ir.utils.prompt import CONTEXTUAL_PROMPT, RAG_PROMPT

load_dotenv()

model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL") or "mistral",
    base_url=os.getenv("OLLAMA_URL") or "http://localhost:11434",
)
embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL") or "mxbai-embed-large",
    base_url=os.getenv("EMBEDDING_URL") or "http://localhost:11434",
)
cross_encoder = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(
        model_name=os.getenv("CROSS_ENCODER_MODEL") or "BAAI/bge-reranker-base",
    ),
    top_n=5,
)
chunker = SemanticChunker(embeddings)
es_client = Elasticsearch(
    [os.getenv("ELASTIC_URL") or "https://localhost:9200"],
    basic_auth=(
        os.getenv("ELASTIC_USER") or "",
        os.getenv("ELASTIC_PASSWORD") or "",
    ),
    ca_certs="./ca.crt",
)
vector_store = ElasticsearchStore(
    embedding=embeddings,
    index_name=os.getenv("ELASTICSEARCH_INDEX") or "langchain",
    es_connection=es_client,
    strategy=HybridRetrievalStrategy(rrf=False),
)

retriever = ContextualCompressionRetriever(
    base_compressor=cross_encoder,
    base_retriever=vector_store.as_retriever(search_kwargs={"k": 25, "fetch_k": 100}),
)


rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextual_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUAL_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def create_index():
    if vector_store.client.indices.exists(index=vector_store.index_name):
        return
    for file in os.listdir("./docs"):
        if not file.endswith(".pdf"):
            continue
        loader = UnstructuredLoader(f"./docs/{file}")
        docs = loader.load()
        docs = list(chunker.transform_documents(docs))
        vector_store.add_documents(docs)


def create_chain():
    def contextual_route(input: dict[str, str]):
        if not input.get("chat_history", False):
            return (lambda x: format_query(x["input"])) | retriever
        return contextual_prompt | model | StrOutputParser() | format_query | retriever

    rag_chain = (
        {
            "context": RunnableLambda(contextual_route) | format_docs,
            "chat_history": lambda x: x["chat_history"],
            "input": lambda x: x["input"],
        }
        | rag_prompt
        | model
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    create_index()
    chain = create_chain()
    chat_history = []

    while True:
        try:
            input_text = input(">>>  ")

            if input_text.strip() == "":
                continue

            ai_answer = ""

            for chunk in chain.stream(
                {"input": input_text, "chat_history": chat_history}
            ):
                ai_answer += chunk
                print(chunk, end="")

            print("\n\n", "-" * 80)

            chat_history.extend(
                [
                    HumanMessage(content=input_text),
                    ai_answer,
                ]
            )
        except KeyboardInterrupt:
            break
