import os

from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests.tools import ToolsIntegrationTests

from langchain_ibm import WatsonxTool

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"

wx_credentials = Credentials(url=URL, api_key=WX_APIKEY)


class TestWatsonxToolsStandard(ToolsIntegrationTests):
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
            "tool_input_schema": {
                "type": "object",
                "properties": {
                    "q": {
                        "title": "Query",
                        "description": "GoogleSearch query",
                        "type": "string",
                    }
                },
                "required": ["q"],
            },
            "watsonx_client": APIClient(
                credentials=wx_credentials,
                project_id=WX_PROJECT_ID,
            ),
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {
            "q": "Search IBM",
        }
