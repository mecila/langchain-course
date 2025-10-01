from typing import Any, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for agent events."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print(f"Chat model ended, response: {response.generations[0][0].text}")
        print("************")

    def on_llm_start(self, serialized: dict, prompts: List[str], **kwargs: Any) -> None:
        print(f"Chat model started with prompts: {prompts}")
        print("************")
