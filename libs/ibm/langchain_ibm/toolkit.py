"""IBM watsonx.ai Toolkit wrapper."""

import urllib.parse
from typing import (
    Dict,
    Optional,
    Type,
    Union,
)

from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
from ibm_watsonx_ai.foundation_models.utils import Tool, Toolkit  # type: ignore
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools.base import BaseTool, BaseToolkit
from langchain_core.utils.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_ibm.utils import check_for_attribute


class ToolSchema(BaseModel):
    input: Union[str, dict]
    """Input to be used when running a tool."""

    config: Optional[dict] = None
    """Configuration options that can be passed for some tools, 
    must match the config schema for the tool."""


class WatsonxTool(BaseTool):
    """IBM watsonx.ai Tool."""

    name: str
    """Name of the tool."""

    description: str
    """Description of what the tool is used for."""

    agent_description: Optional[str] = None
    """The precise instruction to agent LLMs 
    and should be treated as part of the system prompt."""

    tool_input_schema: Optional[Dict] = None
    """Schema of the input that is provided when running the tool if applicable."""

    tool_config_schema: Optional[Dict] = None
    """Schema of the config that can be provided when running the tool if applicable."""

    args_schema: Type[BaseModel] = ToolSchema

    watsonx_tool: Tool = Field(default=None, exclude=True)  #: :meta private:

    watsonx_client: APIClient = Field(exclude=True)

    @model_validator(mode="after")
    def validate_tool(self) -> Self:
        self.watsonx_tool = Tool(
            api_client=self.watsonx_client,
            name=self.name,
            description=self.description,
            agent_description=self.agent_description,
            input_schema=self.tool_input_schema,
            config_schema=self.tool_config_schema,
        )
        return self

    def _run(
        self,
        input: Union[str, dict],
        config: Optional[dict] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Run the tool."""
        return self.watsonx_tool.run(input, config)


class WatsonxToolkit(BaseToolkit):
    """IBM watsonx.ai Toolkit.

    .. dropdown:: Setup
        :open:

        To use, you should have ``langchain_ibm`` python package installed,
        and the environment variable ``WATSONX_APIKEY`` set with your API key, or pass
        it as a named parameter to the constructor.

        .. code-block:: bash

            pip install -U langchain-ibm
            export WATSONX_APIKEY="your-api-key"


    Example:
        .. code-block:: python

            from langchain_ibm import WatsonxToolkit

            watsonx_toolkit = WatsonxToolkit(
                url="https://us-south.ml.cloud.ibm.com",
                apikey="*****",
                project_id="*****",
            )
            tools = watsonx_toolkit.get_tools()

            google_search = watsonx_toolkit.get_tool("GoogleSearch")

            config = {
                "maxResults": 3,
            }
            input = {
                "input": "Search IBM",
                "config": config,
            }
            search_result = google_search.invoke(input=input)

    """

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
                    "Invalid 'url'. Please note that WatsonxToolkit is supported "
                    "only on Cloud and is not yet available for IBM Cloud Pak for Data."
                )

            credentials = Credentials(
                url=self.url.get_secret_value() if self.url else None,
                api_key=self.apikey.get_secret_value() if self.apikey else None,
                token=self.token.get_secret_value() if self.token else None,
                verify=self.verify,
            )
            self.watsonx_client = APIClient(
                credentials=credentials,
                project_id=self.project_id,
                space_id=self.space_id,
            )
            self.watsonx_toolkit = Toolkit(self.watsonx_client)

        return self

    def get_tools(self) -> list[WatsonxTool]:  # type: ignore
        """Get the tools in the toolkit."""
        tools = self.watsonx_toolkit.get_tools()

        return [
            WatsonxTool(
                watsonx_client=self.watsonx_client,
                name=tool["name"],
                description=tool["description"],
                agent_description=tool.get("agent_description"),
                tool_input_schema=tool.get("input_schema"),
                tool_config_schema=tool.get("config_schema"),
            )
            for tool in tools
        ]

    def get_tool(self, tool_name: str) -> WatsonxTool:
        """Get the tool with a given name."""
        tools = self.get_tools()
        for tool in tools:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"A tool with the given name ({tool_name}) was not found.")
