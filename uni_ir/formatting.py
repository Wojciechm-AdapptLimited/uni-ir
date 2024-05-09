from uuid import uuid4
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic.v1 import BaseModel


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
