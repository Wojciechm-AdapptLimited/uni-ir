import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from pydantic.v1 import BaseModel, Field

from uni_ir.formatting import ToolExample

ANALYZE_PROMPT = """You are an expert at converting user questions into database queries. \
You have access to a collection of documents regarding the topic of the user question. \

Perform query decomposition. Given a user question, break it down into distinct sub questions that \
you need to answer in order to answer the original question. Format the sub questions in a simplistic form, using mainly keywords. \

If there are acronyms or words you are not familiar with, do not try to rephrase them. """


class Search(BaseModel):
    """Search over a collection of documents."""

    queries: list[str] = Field(
        ..., description="List of queries against a collection of documents."
    )


class QueryAnalyzer:
    chain: Runnable

    def __init__(self, chain: Runnable):
        self.chain = chain

    @classmethod
    def from_llm(
        cls,
        llm: BaseChatModel,
        *,
        system_prompt: str = ANALYZE_PROMPT,
        examples: list[ToolExample] | None = None,
    ):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("examples", optional=True),
                ("human", "{query}"),
            ]
        )

        function_model = llm.bind(
            functions=[
                {
                    "name": Search.__name__,
                    "description": Search.__doc__,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["queries"],
                    },
                }
            ],
        )

        if examples:
            ex_msgs = [msg for ex in examples for msg in ex.to_messages()]
            prompt = prompt.partial(examples=ex_msgs)

        return cls(chain=prompt | function_model)

    def analyze(
        self,
        query: str,
    ) -> list[str]:
        result = self.chain.invoke(query)
        queries = json.loads(result.additional_kwargs["function_call"]["arguments"])[
            "queries"
        ]
        return queries or [query]
