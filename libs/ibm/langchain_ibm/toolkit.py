"""IBM watsonx.ai Toolkit wrapper."""

import urllib.parse
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
from langchain_core.messages import ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.tools.base import BaseTool, BaseToolkit, _prep_run_args
from langchain_core.utils.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_ibm.utils import check_for_attribute


class ToolSchema(BaseModel):
    """Input for ToolSchema."""

    input: str


class WatsonxTool(BaseTool):
    """IBM watsonx.ai Tool."""

    name: str
    """Name of the tool."""

    description: str
    """Description of what the tool is used for."""

    agent_description: Optional[str] = None
    """The precise instruction to agent LLMs and should be treated as part of the system prompt"""

    # args_schema: Type[BaseModel] = SchemaClass

    def _run(self):
        """Run the tool."""
        pass


class WatsonxToolkit(BaseToolkit):
    """IBM watsonx.ai Toolkit."""

    project_id: Optional[str] = None
    """ID of the watsonx.ai Studio project."""

    space_id: Optional[str] = None
    """ID of the watsonx.ai Studio space."""

    url: SecretStr = Field(
        alias="url",
        default_factory=secret_from_env("WATSONX_URL", default=None),  # type: ignore[assignment]
    )
    """URL to the watsonx.ai Runtime."""

    apikey: Optional[SecretStr] = Field(
        alias="apikey", default_factory=secret_from_env("WATSONX_APIKEY", default=None)
    )
    """API key to the watsonx.ai Runtime."""

    token: Optional[SecretStr] = Field(
        alias="token", default_factory=secret_from_env("WATSONX_TOKEN", default=None)
    )
    """Token to the watsonx.ai Runtime."""

    verify: Union[str, bool, None] = None
    """You can pass one of following as verify:
        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * True - default path to truststore will be taken
        * False - no verification will be made"""

    watsonx_toolkit: Toolkit = Field(default=None, exclude=True)  #: :meta private:

    watsonx_client: Optional[APIClient] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that credentials and python package exists in environment."""
        if isinstance(self.watsonx_client, APIClient):
            self.watsonx_toolkit = Toolkit(self.watsonx_client)
        else:
            check_for_attribute(self.url, "url", "WATSONX_URL")

            parsed_url = urllib.parse.urlparse(self.url.get_secret_value())
            if parsed_url.netloc.endswith(".cloud.ibm.com"):
                if not self.token and not self.apikey:
                    raise ValueError(
                        "Did not find 'apikey' or 'token',"
                        " please add an environment variable"
                        " `WATSONX_APIKEY` or 'WATSONX_TOKEN' "
                        "which contains it,"
                        " or pass 'apikey' or 'token'"
                        " as a named parameter."
                    )
            else:
                raise ValueError(
                    "Did not find '.cloud.ibm.com' in the URL. "
                    "Note that `WatsonxToolkit` is supported only on Cloud."
                )

            credentials = Credentials(
                url=self.url.get_secret_value() if self.url else None,
                api_key=self.apikey.get_secret_value() if self.apikey else None,
                token=self.token.get_secret_value() if self.token else None,
                verify=self.verify,
            )
            api_client = APIClient(
                credentials=credentials,
                project_id=self.project_id,
                space_id=self.space_id,
            )
            self.watsonx_toolkit = Toolkit(api_client)

        return self

    def get_tools(self) -> list[WatsonxTool]:
        """Get the tools in the toolkit."""
        tools = self.watsonx_toolkit.get_tools()

        return [
            WatsonxTool(
                name=tool["name"],
                description=tool["description"],
                agent_description=tool.get("agent_description"),

            )
            for tool in tools
        ]
