import os
from typing import Any

from ibm_watsonx_ai import APIClient, Credentials
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests.tools import ToolsIntegrationTests

from langchain_ibm.agent_toolkits.utility import WatsonxTool

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"

wx_credentials = Credentials(url=URL, api_key=WX_APIKEY)


class TestWatsonxToolsStandard(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool] | BaseTool:
        return WatsonxTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {
            "name": "Weather",
            "description": "Find the weather for a location.",
            "agent_description": "Find the weather for a location.",
            "tool_input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "description": "Name of the location",
                        "type": "string",
                    },
                    "country": {
                        "description": "Name of the state or country",
                        "type": "string",
                    },
                },
                "required": ["location"],
            },
            "watsonx_client": APIClient(
                credentials=wx_credentials,
                project_id=WX_PROJECT_ID,
            ),
        }

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {
            "location": "Cracow",
            "country": "Poland",
        }
