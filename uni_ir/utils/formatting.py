from langchain_core.documents import Document


def pretty_print_docs(docs: list[Document]):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def format_query(query: str) -> str:
    return f"Represent this sentence for searching relevant passages: {query}"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
