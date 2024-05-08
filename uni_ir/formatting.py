from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic.v1 import BaseModel


def pretty_print_docs(docs: list[Document]):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"\nDocument {i+1} | {d.metadata['source']} | {d.metadata['section']}:\n\n{d.page_content}"
                for i, d in enumerate(docs)
            ]
        )
    )


def format_query(query: str) -> str:
    return f"Represent this sentence for searching relevant passages: {query}"


def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


def to_json(docs: list[Document]):
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]


def from_json(docs: list[dict]):
    return [Document(**doc) for doc in docs]


class ToolExample(BaseModel):
    input: str
    calls: list[BaseModel]
    outputs: list[str] | None = None

    def to_messages(self) -> list[BaseMessage]:
        messages: list[BaseMessage] = [HumanMessage(content=self.input)]
        calls = []
        outputs = self.outputs or [
            "This is an example of a correct usage of this tool. Make sure to continue using the tool this way."
        ] * len(self.calls)

        for call in self.calls:
            calls.append(
                {
                    "id": str(uuid4()),
                    "function": {
                        "name": call.__class__.__name__,
                        "arguments": call.json(),
                    },
                }
            )
        messages.append(AIMessage(content="", additional_kwargs={"tool_calls": calls}))

        for call, output in zip(calls, outputs):
            messages.append(ToolMessage(tool_call_id=call["id"], content=output))

        return messages
