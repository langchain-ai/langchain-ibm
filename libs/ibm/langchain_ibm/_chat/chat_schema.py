from collections.abc import Callable
from typing import List, Optional, Union

from langchain_core.messages import ToolCall
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel


class ChatSchema(BaseModel):
    model_id: str
    """The Watsonx.ai model ID"""

    prompt_template: PromptTemplate
    """
    Jinja2 prompt template used for formatting messages into a correctly
    formatted prompt for the model. At instantiation time, a list of `BaseMessage`
    objects is passed to the template, along with available tools (if supported by the model).
    """

    tools: bool
    """Whether or not this model supports tools."""

    tools_parser: Optional[Callable[[str], Union[str, List[ToolCall]]]] = None
    """
    Function to parse tool results from the model output. Must be provided if `tools` is True.
    This function will be invoked if the model is called with tools enabled (passing "toools" or
    binding them using `bind_tools`) and should return either a List of tool calls if the output
    of the model indicates that tools should be called (as per model's spec), or a string if the
    output is not a tool call.
    """

    tools_stop_sequences: Optional[List[str]] = None
    """Optional additional stop sequences to add to inference parameters if tool use is enabled."""
