"""IBM watsonx.ai Toolkit wrapper."""

from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models.utils import Toolkit
from langchain_core.tools.base import BaseTool, BaseToolkit, _prep_run_args
from langchain_core.messages import ToolCall
from langchain_core.runnables import RunnableConfig


class WatsonxToolkit(BaseToolkit):

    def get_tools(self) -> list[BaseTool]:
        """Get the tools in the toolkit."""


class WatsonxTool(BaseTool):
    def invoke(
            self,
            input: Union[str, dict, ToolCall],
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
    ) -> Any:
        tool_input, kwargs = _prep_run_args(input, config, **kwargs)
        return self.run(tool_input, **kwargs)
