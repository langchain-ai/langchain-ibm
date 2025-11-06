from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.service_instance import (
    ServiceInstance,
)
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests.tools import ToolsUnitTests

from langchain_ibm.agent_toolkits.utility.toolkit import WatsonxTool

client = APIClient.__new__(APIClient)
client.CLOUD_PLATFORM_SPACES = True
client.ICP_PLATFORM_SPACES = True
credentials = Credentials(api_key="api_key")
client.credentials = credentials
client.service_instance = ServiceInstance.__new__(ServiceInstance)


class TestWatsonxToolsStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool] | BaseTool:
        return WatsonxTool

    @property
    def tool_constructor_params(self) -> dict:
        return {
            "name": "GoogleSearch",
            "description": "Search for online trends, news, current events, "
            "real-time information, or research topics.",
            "agent_description": "Search for online trends, news, current events, "
            "real-time information, or research topics.",
            "tool_config_schema": {
                "title": "config schema for GoogleSearch tool",
                "type": "object",
                "properties": {
                    "maxResults": {
                        "title": "Max number of results to return",
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                    }
                },
            },
            "watsonx_client": client,
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {
            "input": "Search IBM",
        }
