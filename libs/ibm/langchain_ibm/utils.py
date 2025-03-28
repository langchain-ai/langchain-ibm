from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from ibm_watsonx_ai.foundation_models.schema import BaseSchema  # type: ignore
from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain_ibm.toolkit import WatsonxTool


def check_for_attribute(value: SecretStr | None, key: str, env_key: str) -> None:
    if not value or not value.get_secret_value():
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )


def extract_params(
    kwargs: Dict[str, Any],
    default_params: Optional[Union[BaseSchema, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if kwargs.get("params") is not None:
        params = kwargs.pop("params")
    elif default_params is not None:
        params = default_params
    else:
        params = None

    if isinstance(params, BaseSchema):
        params = params.to_dict()

    return params or {}


def extract_chat_params(
    kwargs: Dict[str, Any],
    default_params: Optional[Union[BaseSchema, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if kwargs.get("params") is not None:
        params = kwargs.pop("params")
        check_duplicate_chat_params(params, kwargs)
    elif default_params is not None:
        params = deepcopy(default_params)
    else:
        params = None

    if isinstance(params, BaseSchema):
        params = params.to_dict()

    return params or {}


def check_duplicate_chat_params(params: dict, kwargs: dict) -> None:
    duplicate_keys = {k for k, v in kwargs.items() if v is not None and k in params}

    if duplicate_keys:
        raise ValueError(
            f"Duplicate parameters found in params and keyword arguments: "
            f"{list(duplicate_keys)}"
        )


def convert_to_watsonx_tool(tool: "WatsonxTool") -> dict:
    """Convert `WatsonxTool` to watsonx tool structure.

    Args:
        tool: `WatsonxTool` from `WatsonxToolkit`


    Example:

    .. code-block:: python

        from langchain_ibm import WatsonxToolkit

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

    def parse_parameters(input_schema: dict | None) -> dict:
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
