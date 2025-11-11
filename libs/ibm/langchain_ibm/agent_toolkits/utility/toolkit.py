"""IBM watsonx.ai Toolkit wrapper."""

from typing import Any, cast

from ibm_watsonx_ai import APIClient  # type: ignore[import-untyped]
from ibm_watsonx_ai.foundation_models.utils import (  # type: ignore[import-untyped]
    Tool,
    Toolkit,
)
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools.base import BaseTool, BaseToolkit
from langchain_core.utils.utils import secret_from_env
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    create_model,
    model_validator,
)
from typing_extensions import Self, override

from langchain_ibm.agent_toolkits.utility.utils import convert_to_watsonx_tool
from langchain_ibm.utils import (
    normalize_api_key,
    resolve_watsonx_credentials,
    secret_from_env_multi,
)


class WatsonxTool(BaseTool):
    """IBM watsonx.ai Tool."""

    name: str
    """Name of the tool."""

    description: str
    """Description of what the tool is used for."""

    agent_description: str | None = None
    """The precise instruction to agent LLMs
    and should be treated as part of the system prompt."""

    tool_input_schema: dict[str, Any] | None = None
    """Schema of the input that is provided when running the tool if applicable."""

    tool_config_schema: dict[str, Any] | None = None
    """Schema of the config that can be provided when running the tool if applicable."""

    tool_config: dict[str, Any] | None = None
    """Config properties to be used when running a tool if applicable."""

    args_schema: type[BaseModel] = BaseModel

    _watsonx_tool: Tool  #: :meta private:

    watsonx_client: APIClient = Field(exclude=True)

    @model_validator(mode="after")
    def validate_tool(self) -> Self:
        """Validate tool."""
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
            name="ToolArgsSchema",
            schema=json_schema,
        )

        return self

    @override
    def _run(
        self,
        *args: Any,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the tool."""
        if self.tool_input_schema is None:
            input_data = kwargs.get("input") or args[0]
        else:
            input_data = {
                k: v
                for k, v in kwargs.items()
                if k in self.tool_input_schema["properties"]
            }

        return self._watsonx_tool.run(input_data, self.tool_config)

    def set_tool_config(self, tool_config: dict[str, Any]) -> None:
        """Set tool config properties.

        ???+ example "Example"

            ```python
            google_search = watsonx_toolkit.get_tool("GoogleSearch")
            print(google_search.tool_config_schema)
            tool_config = {"maxResults": 3}
            google_search.set_tool_config(tool_config)
            ```

        """
        self.tool_config = tool_config


class WatsonxToolkit(BaseToolkit):
    """IBM watsonx.ai Toolkit.

    ???+ info "Setup"

        To use, you should have `langchain_ibm` python package installed,
        and the environment variable `WATSONX_API_KEY` set with your API key, or pass
        it as a named parameter `api_key` to the constructor.

        ```bash
        pip install -U langchain-ibm

        # or using uv
        uv add langchain-ibm
        ```

        ```bash
        export WATSONX_API_KEY="your-api-key"
        ```

        !!! deprecated
            `apikey` and `WATSONX_APIKEY` are deprecated and will be removed in
            version `2.0.0`. Use `api_key` and `WATSONX_API_KEY` instead.

    ??? info "Instantiate"

        IBM watsonx.ai for IBM Cloud:

        ```python
        from langchain_ibm.agent_toolkits.utility import WatsonxToolkit

        watsonx_toolkit = WatsonxToolkit(
            url="https://us-south.ml.cloud.ibm.com",
            project_id="*****",  # or `space_id`
            api_key="*****",  # not needed if `WATSONX_API_KEY` is set
        )
        ```

        IBM watsonx.ai software:
        ```python
        from langchain_ibm.agent_toolkits.utility import WatsonxToolkit

        watsonx_toolkit = WatsonxToolkit(
            url="<CPD_URL>",
            project_id="*****",  # or `space_id`
            username="*****",
            password="*****",
            instance_id="*****",
            version="*****",  # optional
        )
        ```

    ??? info "Invoke"

        ```python
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
        ```

    ??? info "Run"

        ```python
        rag_query = watsonx_toolkit.get_tool(tool_name="RAGQuery")

        rag_query.set_tool_config(
            {
                "vectorIndexId": "<vector-index-id>",
                "projectId": "<project-id>",
            }
        )

        res = rag_query.run("How to initialize APIClient?")
        ```

    """

    project_id: str | None = None
    """ID of the watsonx.ai Studio project."""

    space_id: str | None = None
    """ID of the watsonx.ai Studio space."""

    url: SecretStr = Field(
        alias="url",
        default_factory=secret_from_env("WATSONX_URL", default=None),  # type: ignore[assignment]
    )
    """URL to the watsonx.ai Runtime."""

    apikey: SecretStr | None = None
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env_multi(
            names_priority=["WATSONX_API_KEY", "WATSONX_APIKEY"],
            deprecated={"WATSONX_APIKEY"},
        ),
        serialization_alias="api_key",
        validation_alias=AliasChoices("api_key", "apikey"),  # accept both on input
        description="API key to the Watson Machine Learning or CPD instance.",
    )
    """API key to the watsonx.ai Runtime."""

    token: SecretStr | None = Field(
        alias="token",
        default_factory=secret_from_env("WATSONX_TOKEN", default=None),
    )
    """Token to the watsonx.ai Runtime."""

    password: SecretStr | None = Field(
        alias="password",
        default_factory=secret_from_env("WATSONX_PASSWORD", default=None),
    )
    """Password to the CPD instance."""

    username: SecretStr | None = Field(
        alias="username",
        default_factory=secret_from_env("WATSONX_USERNAME", default=None),
    )
    """Username to the CPD instance."""

    instance_id: SecretStr | None = Field(
        alias="instance_id",
        default_factory=secret_from_env("WATSONX_INSTANCE_ID", default=None),
    )
    """Instance_id of the CPD instance."""

    version: SecretStr | None = None
    """Version of the CPD instance."""

    verify: str | bool | None = None
    """You can pass one of following as verify:
        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * True - default path to truststore will be taken
        * False - no verification will be made"""

    _tools: list[WatsonxTool] | None = None
    """Tools in the toolkit."""

    _watsonx_toolkit: Toolkit | None = PrivateAttr(default=None)  #: :meta private:

    watsonx_client: APIClient | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_and_warn_deprecated_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Handle deprecated input kwarg name `apikey` vs new `api_key`.
            data = normalize_api_key(data=data)
        return data

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that credentials and python package exists in environment."""
        if isinstance(self.watsonx_client, APIClient):
            self._watsonx_toolkit = Toolkit(self.watsonx_client)
        else:
            credentials = resolve_watsonx_credentials(
                url=self.url,
                api_key=self.api_key,
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

    def get_tools(self) -> list[WatsonxTool]:  # type: ignore[override]
        """Get the tools in the toolkit."""
        return self._tools  # type: ignore[return-value]

    def get_tool(self, tool_name: str) -> WatsonxTool:
        """Get the tool with a given name."""
        for tool in self.get_tools():
            if tool.name == tool_name:
                return tool
        error_msg = f"A tool with the given name ({tool_name}) was not found."
        raise ValueError(error_msg)


def _json_schema_to_pydantic_model(
    name: str,
    schema: dict[str, Any],
) -> type[BaseModel]:
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

    return cast("type[BaseModel]", create_model(name, **fields))  # type: ignore[call-overload]
