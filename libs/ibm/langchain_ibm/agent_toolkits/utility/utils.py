from copy import deepcopy
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langchain_ibm.agent_toolkits.utility.toolkit import WatsonxTool


def convert_to_watsonx_tool(tool: "WatsonxTool") -> dict:
    """Convert `WatsonxTool` to watsonx tool structure.

    Args:
        tool: `WatsonxTool` from `WatsonxToolkit`


    Example:

    .. code-block:: python

        from langchain_ibm.agents_toolkits.utility import WatsonxToolkit

        watsonx_toolkit = WatsonxToolkit(
            url="https://us-south.ml.cloud.ibm.com",
            apikey="*****",
        )
        weather_tool = watsonx_toolkit.get_tool("Weather")
        convert_to_watsonx_tool(weather_tool)

        # Return
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "Weather",
        #         "description": "Find the weather for a city.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "location": {
        #                     "title": "location",
        #                     "description": "Name of the location",
        #                     "type": "string",
        #                 },
        #                 "country": {
        #                     "title": "country",
        #                     "description": "Name of the state or country",
        #                     "type": "string",
        #                 },
        #             },
        #             "required": ["location"],
        #         },
        #     },
        # }

    """

    def parse_parameters(input_schema: Optional[dict]) -> dict:
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

    watsonx_tool = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parse_parameters(tool.tool_input_schema),
        },
    }
    return watsonx_tool
