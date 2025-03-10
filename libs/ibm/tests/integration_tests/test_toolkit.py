"""Test WatsonxToolkit.

You'll need to set WATSONX_APIKEY and WATSONX_PROJECT_ID environment variables.
"""

import json
import os

from langchain_ibm import WatsonxToolkit

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"

TOOL_NAME_1 = "GoogleSearch"
TOOL_NAME_2 = "Weather"


def test_01_get_tools() -> None:
    watsonx_toolkit = WatsonxToolkit(
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
    )
    tools = watsonx_toolkit.get_tools()
    assert tools


def test_02_get_tool_with_config_schema() -> None:
    watsonx_toolkit = WatsonxToolkit(
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
    )
    tool = watsonx_toolkit.get_tool(tool_name=TOOL_NAME_1)
    assert tool.name == TOOL_NAME_1
    assert tool.description
    assert tool.tool_config_schema


def test_03_get_tool_with_input_schema() -> None:
    watsonx_toolkit = WatsonxToolkit(
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
    )
    tool = watsonx_toolkit.get_tool(tool_name=TOOL_NAME_2)
    assert tool.name == TOOL_NAME_2
    assert tool.description
    assert tool.tool_input_schema


def test_04_invoke_tool_with_config_schema() -> None:
    watsonx_toolkit = WatsonxToolkit(
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
    )
    tool = watsonx_toolkit.get_tool(tool_name=TOOL_NAME_1)
    assert tool.name == TOOL_NAME_1
    assert tool.tool_config_schema

    config = {
        "maxResults": 3,
    }
    tool_input = {
        "input": "Search IBM",
        "config": config,
    }

    result = tool.invoke(tool_input)
    output = json.loads(result.get("output"))

    assert isinstance(output, list)
    assert len(output) == 3
    for answer in output:
        assert answer.get("title")
        assert answer.get("description")
        assert answer.get("url")


def test_05_invoke_tool_with_input_schema() -> None:
    watsonx_toolkit = WatsonxToolkit(
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
    )
    tool = watsonx_toolkit.get_tool(tool_name=TOOL_NAME_2)
    assert tool.name == TOOL_NAME_2
    assert tool.tool_input_schema

    tool_input = {
        "input": {
            "name": "New York",
        },
    }

    result = tool.invoke(tool_input)
    output = result.get("output")

    assert output
    assert tool_input["input"]["name"] in output
    assert "temperature" in output.lower()
