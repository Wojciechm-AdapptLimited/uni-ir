import os
from dotenv import load_dotenv

from elasticsearch import Elasticsearch
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from unstructured.partition.auto import detect_filetype, FileType

from uni_ir.reranker import CrossEncoderReranker
from uni_ir.load import ImageCaptioner, DocumentLoader
from uni_ir.utils.formatting import format_docs, format_query, pretty_print_docs
from uni_ir.utils.prompt import CONTEXTUAL_PROMPT, RAG_PROMPT

load_dotenv()

# model = ChatOllama(
#     model=os.getenv("OLLAMA_MODEL") or "mixtral:8x7b-instruct-v0.1-q2_K",
#     base_url=os.getenv("OLLAMA_URL") or "http://localhost:11434",
# )
embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL") or "mxbai-embed-large",
    base_url=os.getenv("EMBEDDING_URL") or "http://localhost:11434",
)
# cross_encoder = CrossEncoderReranker(
#     model=HuggingFaceCrossEncoder(
#         model_name=os.getenv("CROSS_ENCODER_MODEL") or "BAAI/bge-reranker-v2-m3",
#     ),
#     threshold=0.5,
# )
captioning_processor = Blip2Processor.from_pretrained(
    os.getenv("CAPTIONING_MODEL") or "Salesforce/blip2-opt-2.7b"
)
captioning_model = Blip2ForConditionalGeneration.from_pretrained(
    os.getenv("CAPTIONING_MODEL") or "Salesforce/blip2-opt-2.7b"
)

if isinstance(captioning_processor, tuple):
    captioning_processor = captioning_processor[0]

if not isinstance(captioning_model, Blip2ForConditionalGeneration):
    raise ValueError("Invalid captioning model")

image_captioner = ImageCaptioner(captioning_processor, captioning_model)
# image_captioner = ImageCaptioner(model=captioning_model, processor=captioning_processor)
document_loader = DocumentLoader(embeddings, image_captioner)

# es_client = Elasticsearch(
#     [os.getenv("ELASTIC_URL") or "https://localhost:9200"],
#     basic_auth=(
#         os.getenv("ELASTIC_USER") or "",
#         os.getenv("ELASTIC_PASSWORD") or "",
#     ),
#     ca_certs="./ca.crt",
# )
# vector_store = ElasticsearchStore(
#     embedding=embeddings,
#     index_name=os.getenv("ELASTICSEARCH_INDEX") or "langchain",
#     es_connection=es_client,
#     strategy=HybridRetrievalStrategy(rrf=False),
# )
#
# retriever = ContextualCompressionRetriever(
#     base_compressor=cross_encoder,
#     base_retriever=vector_store.as_retriever(search_kwargs={"k": 25, "fetch_k": 100}),
# )
#
#
# rag_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", RAG_PROMPT),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{query}"),
#     ]
# )
#
# contextual_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", CONTEXTUAL_PROMPT),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{query}"),
#     ]
# )


def create_index():
    # if vector_store.client.indices.exists(index=vector_store.index_name):
    # return
    for filename in os.listdir("./docs"):
        filetype = detect_filetype(filename)

        if filetype in [FileType.UNK, FileType.EMPTY]:
            continue

        print(f"Loading {filename}...")
        with open(f"./docs/{filename}", "rb") as file:
            docs = document_loader.load(file, filename)

        pretty_print_docs(docs)
        # vector_store.add_documents(list(docs))


# def create_chain():
#     def contextual_route(input: dict[str, str]):
#         if not input.get("chat_history", False):
#             return (lambda x: format_query(x["query"])) | retriever
#         return contextual_prompt | model | StrOutputParser() | format_query | retriever
#
#     rag_chain = (
#         {
#             "context": lambda x: format_docs(x["context"]),
#             "query": lambda x: x["input"]["query"],
#             "chat_history": lambda x: x["input"]["chat_history"],
#         }
#         | rag_prompt
#         | model
#         | StrOutputParser()
#     )
#
#     contextual_chain = RunnableParallel(
#         {
#             "context": RunnableLambda(contextual_route),
#             "input": RunnablePassthrough(),
#         }
#     ).assign(answer=rag_chain)
#
#     return contextual_chain


def main():
    pass


if __name__ == "__main__":
    create_index()
    main()
    # create_index()
    # chain = create_chain()
    # chat_history = []
    #
    # while True:
    #     try:
    #         input_text = input(">>>  ")
    #
    #         if input_text.strip() == "":
    #             continue
    #
    #         ai_answer = ""
    #         context: list[Document] | None = None
    #
    #         for chunk in chain.stream(
    #             {"query": input_text, "chat_history": chat_history}
    #         ):
    #             partial_answer = chunk.get("answer", "")
    #             context = context or chunk.get("context", None)
    #
    #             ai_answer += partial_answer
    #
    #             print(partial_answer, end="", flush=True)
    #
    #         # if context:
    #         #     print("\n\n\n")
    #         #     pretty_print_docs(context)
    #         #
    #         print("\n\n", "-" * 80)
    #
    #         chat_history.extend(
    #             [
    #                 HumanMessage(content=input_text),
    #                 ai_answer,
    #             ]
    #         )
    #     except KeyboardInterrupt:
    #         break
