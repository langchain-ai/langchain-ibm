"""IBM watsonx.ai Toolkit wrapper."""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from ibm_watsonx_ai import APIClient  # type: ignore
from ibm_watsonx_ai.foundation_models.utils import (  # type: ignore
    Tool,
    Toolkit,
)
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools.base import BaseTool, BaseToolkit
from langchain_core.utils.utils import secret_from_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    create_model,
    model_validator,
)
from typing_extensions import Self

from langchain_ibm.utils import resolve_watsonx_credentials

from .utils import convert_to_watsonx_tool


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

    tool_config: Optional[Dict] = None
    """Config properties to be used when running a tool if applicable."""

    args_schema: Type[BaseModel] = BaseModel

    _watsonx_tool: Optional[Tool] = PrivateAttr(default=None)  #: :meta private:

    watsonx_client: APIClient = Field(exclude=True)

    @model_validator(mode="after")
    def validate_tool(self) -> Self:
        self._watsonx_tool = Tool(
            api_client=self.watsonx_client,
            name=self.name,
            description=self.description,
            agent_description=self.agent_description,
            input_schema=self.tool_input_schema,
            config_schema=self.tool_config_schema,
        )
        converted_tool = convert_to_watsonx_tool(self)
        json_schema = converted_tool["function"]["parameters"]
        self.args_schema = _json_schema_to_pydantic_model(
            name="ToolArgsSchema", schema=json_schema
        )

        return self

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict:
        """Run the tool."""
        if self.tool_input_schema is None:
            input = kwargs.get("input") or args[0]
        else:
            input = {
                k: v
                for k, v in kwargs.items()
                if k in self.tool_input_schema["properties"]
            }

        return self._watsonx_tool.run(input, self.tool_config)  # type: ignore[union-attr]

    def set_tool_config(self, tool_config: dict) -> None:
        """Set tool config properties.

        Example:
        .. code-block:: python

            google_search = watsonx_toolkit.get_tool("GoogleSearch")
            print(google_search.tool_config_schema)
            tool_config = {
                "maxResults": 3
            }
            google_search.set_tool_config(tool_config)

        """
        self.tool_config = tool_config


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


    IBM watsonx.ai for IBM Cloud example:
        .. code-block:: python

            from langchain_ibm.agent_toolkits.utility import WatsonxToolkit

            watsonx_toolkit = WatsonxToolkit(
                url="https://us-south.ml.cloud.ibm.com",
                apikey="*****",
            )
            tools = watsonx_toolkit.get_tools()

            google_search = watsonx_toolkit.get_tool(tool_name="GoogleSearch")

            tool_config = {
                "maxResults": 3,
            }
            google_search.set_tool_config(tool_config)
            input = {
                "input": "Search IBM",
            }
            search_result = google_search.invoke(input)

    IBM watsonx.ai software example:
        .. code-block:: python

            from langchain_ibm.agent_toolkits.utility import WatsonxToolkit

            watsonx_toolkit = WatsonxToolkit(
                url="<CPD_URL>",
                username="*****",
                password="*****",
                instance_id="*****",
                version="*****"  # optional
            )

            rag_query = watsonx_toolkit.get_tool(tool_name="RAGQuery")

            rag_query.set_tool_config(
                {
                    "vectorIndexId": "<vector-index-id>",
                    "projectId": "<project-id>",
                }
            )

            res = rag_query.run("How to initialize APIClient?")

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

    password: Optional[SecretStr] = Field(
        alias="password",
        default_factory=secret_from_env("WATSONX_PASSWORD", default=None),
    )
    """Password to the CPD instance."""

    username: Optional[SecretStr] = Field(
        alias="username",
        default_factory=secret_from_env("WATSONX_USERNAME", default=None),
    )
    """Username to the CPD instance."""

    instance_id: Optional[SecretStr] = Field(
        alias="instance_id",
        default_factory=secret_from_env("WATSONX_INSTANCE_ID", default=None),
    )
    """Instance_id of the CPD instance."""

    version: Optional[SecretStr] = None
    """Version of the CPD instance."""

    verify: Union[str, bool, None] = None
    """You can pass one of following as verify:
        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * True - default path to truststore will be taken
        * False - no verification will be made"""

    _tools: Optional[List[WatsonxTool]] = None
    """Tools in the toolkit."""

    _watsonx_toolkit: Optional[Toolkit] = PrivateAttr(default=None)  #: :meta private:

    watsonx_client: Optional[APIClient] = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that credentials and python package exists in environment."""
        if isinstance(self.watsonx_client, APIClient):
            self._watsonx_toolkit = Toolkit(self.watsonx_client)
        else:
            credentials = resolve_watsonx_credentials(
                url=self.url,
                apikey=self.apikey,
                token=self.token,
                password=self.password,
                username=self.username,
                instance_id=self.instance_id,
                version=self.version,
                verify=self.verify,
            )
            self.watsonx_client = APIClient(
                credentials=credentials,
                project_id=self.project_id,
                space_id=self.space_id,
            )
            self._watsonx_toolkit = Toolkit(self.watsonx_client)

        self._tools = [
            WatsonxTool(
                watsonx_client=self.watsonx_client,
                name=tool["name"],
                description=tool["description"],
                agent_description=tool.get("agent_description"),
                tool_input_schema=tool.get("input_schema"),
                tool_config_schema=tool.get("config_schema"),
            )
            for tool in self._watsonx_toolkit.get_tools()
        ]

        return self

    def get_tools(self) -> list[WatsonxTool]:  # type: ignore
        """Get the tools in the toolkit."""
        return self._tools  # type: ignore[return-value]

    def get_tool(self, tool_name: str) -> WatsonxTool:
        """Get the tool with a given name."""
        for tool in self.get_tools():
            if tool.name == tool_name:
                return tool
        raise ValueError(f"A tool with the given name ({tool_name}) was not found.")


def _json_schema_to_pydantic_model(
    name: str, schema: Dict[str, Any]
) -> Type[BaseModel]:
    properties = schema.get("properties", {})
    fields = {}

    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for field_name, field_schema in properties.items():
        field_type = field_schema.get("type", "string")
        is_required = field_name in schema.get("required", [])

        py_type = type_mapping.get(field_type, Any)

        fields[field_name] = (py_type, ... if is_required else None)

    return create_model(name, **fields)  # type: ignore[call-overload]
