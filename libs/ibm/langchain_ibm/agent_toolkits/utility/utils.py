"""Utility helpers."""

from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_ibm.agent_toolkits.utility.toolkit import WatsonxTool


def convert_to_watsonx_tool(tool: "WatsonxTool") -> dict[str, Any]:
    """Convert `WatsonxTool` to watsonx tool structure.

    Args:
        tool: `WatsonxTool` from `WatsonxToolkit`

    ???+ example "Example"

        ```python
        from langchain_ibm.agents_toolkits.utility import WatsonxToolkit

        watsonx_toolkit = WatsonxToolkit(
            url="https://us-south.ml.cloud.ibm.com",
            api_key="*****",
        )
        weather_tool = watsonx_toolkit.get_tool("Weather")
        convert_to_watsonx_tool(weather_tool)
        ```

        ```json
        {
            "type": "function",
            "function": {
                "name": "Weather",
                "description": "Find the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "title": "location",
                            "description": "Name of the location",
                            "type": "string",
                        },
                        "country": {
                            "title": "country",
                            "description": "Name of the state or country",
                            "type": "string",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
        ```
    """

    def parse_parameters(input_schema: dict[str, Any] | None) -> dict[str, Any]:
        if input_schema:
            parameters = deepcopy(input_schema)
        else:
            parameters = {
                "type": "object",
                "properties": {
                    "input": {
                        "description": "Input to be used when running tool.",
                        "type": "string",
                    },
                },
                "required": ["input"],
            }

        return parameters

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parse_parameters(tool.tool_input_schema),
        },
    }
