"""Test ChatWatsonx API wrapper."""

import json
import os
from typing import Any
from unittest.mock import Mock

import pytest
from ibm_watsonx_ai import APIClient  # type: ignore
from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
from ibm_watsonx_ai.foundation_models.schema import (  # type: ignore[import-untyped]
    TextChatParameters,
)
from ibm_watsonx_ai.gateway import Gateway  # type: ignore
from ibm_watsonx_ai.wml_client_error import WMLClientError  # type: ignore

from langchain_ibm import ChatWatsonx
from langchain_ibm.chat_models import normalize_tool_arguments

os.environ.pop("WATSONX_APIKEY", None)
os.environ.pop("WATSONX_PROJECT_ID", None)

MODEL_ID = "mistralai/mixtral-8x7b-instruct-v01"

api_client_mock = Mock(spec=APIClient)
api_client_mock.default_space_id = None
api_client_mock.default_project_id = "fake_project_id"

gateway_mock = Mock(spec=Gateway)
gateway_mock._client = api_client_mock

model_inference_mock = Mock(spec=ModelInference)
model_inference_mock._client = api_client_mock
model_inference_mock.model_id = "fake_model_id"
model_inference_mock.params = {"temperature": 1}


def test_initialize_chat_watsonx_bad_path_without_url() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(
            model_id=MODEL_ID,
        )

    assert "url" in str(e.value)
    assert "WATSONX_URL" in str(e.value)


def test_initialize_chat_watsonx_cloud_bad_path() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(model_id=MODEL_ID, url="https://us-south.ml.cloud.ibm.com")

    assert "apikey" in str(e.value) and "token" in str(e.value)
    assert "WATSONX_APIKEY" in str(e.value) and "WATSONX_TOKEN" in str(e.value)


def test_initialize_chat_watsonx_cpd_bad_path_without_all() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
        )
    assert (
        "apikey" in str(e.value)
        and "password" in str(e.value)
        and "token" in str(e.value)
    )
    assert (
        "WATSONX_APIKEY" in str(e.value)
        and "WATSONX_PASSWORD" in str(e.value)
        and "WATSONX_TOKEN" in str(e.value)
    )


def test_initialize_chat_watsonx_cpd_bad_path_only_password() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            password="fake_password",
        )
    assert "username" in str(e.value)
    assert "WATSONX_USERNAME" in str(e.value)


def test_initialize_chat_watsonx_cpd_bad_path_only_username() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            username="fake_username",
        )
    assert "password" in str(e.value)
    assert "WATSONX_PASSWORD" in str(e.value)
    assert "apikey" in str(e.value)
    assert "WATSONX_APIKEY" in str(e.value)


def test_initialize_chat_watsonx_cpd_bad_path_only_apikey() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            apikey="fake_apikey",
        )
    assert "username" in str(e.value)
    assert "WATSONX_USERNAME" in str(e.value)


def test_initialize_chat_watsonx_cpd_deprecation_warning_with_instance_id() -> None:
    with (
        pytest.warns(
            DeprecationWarning, match="The `instance_id` parameter is deprecated"
        ),
        pytest.raises(WMLClientError),
    ):
        ChatWatsonx(
            model_id="google/flan-ul2",
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            apikey="test_apikey",
            username="test_user",
            instance_id="openshift",
        )


def test_initialize_chat_watsonx_with_two_exclusive_parameters() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(
            model_id=MODEL_ID,
            model=MODEL_ID,
            url="https://us-south.ml.cloud.ibm.com",
            apikey="test_apikey",
        )

    assert (
        "The parameters 'model', 'model_id' and 'deployment_id' are mutually exclusive."
        " Please specify exactly one of these parameters when initializing ChatWatsonx."
        in str(e.value)
    )


def test_initialize_chat_watsonx_with_three_exclusive_parameters() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(
            model_id=MODEL_ID,
            model=MODEL_ID,
            deployment_id="test_deployment_id",
            url="https://us-south.ml.cloud.ibm.com",
            apikey="test_apikey",
        )

    assert (
        "The parameters 'model', 'model_id' and 'deployment_id' are mutually exclusive."
        " Please specify exactly one of these parameters when initializing ChatWatsonx."
        in str(e.value)
    )


def test_initialize_chat_watsonx_with_api_client_only() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(watsonx_client=api_client_mock)
    assert (
        "The parameters 'model', 'model_id' and 'deployment_id' are mutually exclusive."
        " Please specify exactly one of these parameters when initializing ChatWatsonx."
        in str(e.value)
    )


def test_initialize_chat_watsonx_with_watsonx_model_gateway() -> None:
    with pytest.raises(NotImplementedError) as e:
        ChatWatsonx(watsonx_model_gateway=gateway_mock)
    assert (
        "Passing the 'watsonx_model_gateway' parameter to the ChatWatsonx "
        "constructor is not supported yet." in str(e.value)
    )


def test_initialize_chat_watsonx_without_any_params() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx()
    assert (
        "The parameters 'model', 'model_id' and 'deployment_id' are mutually exclusive."
        " Please specify exactly one of these parameters when initializing ChatWatsonx."
        in str(e.value)
    )


def test_initialize_chat_watsonx_with_model_inference_only() -> None:
    chat = ChatWatsonx(watsonx_model=model_inference_mock)

    assert isinstance(chat, ChatWatsonx)


def test_initialize_chat_watsonx_with_all_supported_params(mocker: Any) -> None:
    top_p = 0.8

    def mock_modelinference_chat(**kwargs: Any) -> dict:
        assert kwargs.get("messages") == [{"content": "Hello", "role": "user"}]
        assert kwargs.get("params") == (
            {
                k: v
                for k, v in TextChatParameters.get_sample_params().items()
                if "guided" not in k
                if "chat_template_kwargs" not in k
                if "reasoning_effort" not in k
                if "include_reasoning" not in k
            }
            | {
                "logit_bias": {"1003": -100, "1004": -100},
                "seed": 41,
                "stop": ["this", "the"],
            }
            | {"top_p": top_p}
        )

        return {"id": "123", "choices": [{"message": {"content": "Hi", "role": "ai"}}]}

    mocker.patch(
        "ibm_watsonx_ai.foundation_models.ModelInference.__init__",
        return_value=None,
    )
    mocker.patch(
        "ibm_watsonx_ai.foundation_models.ModelInference.chat",
        side_effect=mock_modelinference_chat,
    )

    chat = ChatWatsonx(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        apikey="test_apikey",
        frequency_penalty=0.5,
        logprobs=True,
        top_logprobs=3,
        presence_penalty=0.3,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Sample JSON schema",
                "schema": {
                    "title": "SimpleUser",
                    "type": "object",
                    "properties": {
                        "username": {"type": "string"},
                        "email": {"type": "string", "format": "email"},
                    },
                    "required": ["username", "email"],
                },
                "strict": False,
            },
        },
        temperature=0.7,
        max_completion_tokens=512,
        time_limit=600000,
        top_p=0.9,
        n=1,
        logit_bias={"1003": -100, "1004": -100},
        seed=41,
        stop=["this", "the"],
    )

    # change only top_n
    chat.invoke("Hello", top_p=top_p)


# Tests for normalize_tool_arguments function
def test_normalize_tool_arguments_with_json_string() -> None:
    """Test normalizing JSON string arguments."""
    # Test case 1: JSON string
    json_str = '{"location": "San Francisco", "unit": "celsius"}'
    result = normalize_tool_arguments(json_str)
    assert result == '{"location": "San Francisco", "unit": "celsius"}'


def test_normalize_tool_arguments_with_python_dict_string() -> None:
    """Test normalizing Python dict string arguments."""
    # Test case 2: Python dict string with single quotes
    python_dict_str = "{'location': 'San Francisco', 'unit': 'celsius'}"
    result = normalize_tool_arguments(python_dict_str)
    assert result == '{"location": "San Francisco", "unit": "celsius"}'


def test_normalize_tool_arguments_with_extra_quotes() -> None:
    """Test normalizing arguments with extra surrounding quotes."""
    # Test case 3: Extra wrapping quotes like '"{...}"'
    wrapped_json = '"{\\"location\\": \\"San Francisco\\", \\"unit\\": \\"celsius\\"}"'
    result = normalize_tool_arguments(wrapped_json)
    assert result == '{"location": "San Francisco", "unit": "celsius"}'


def test_normalize_tool_arguments_with_nested_structures() -> None:
    """Test normalizing arguments with nested dict/list structures."""
    # Test case 4: Nested structures
    nested_str = '{"user": {"name": "John", "prefs": ["temp", "humidity"]}}'
    result = normalize_tool_arguments(nested_str)
    assert result == '{"user": {"name": "John", "prefs": ["temp", "humidity"]}}'


def test_normalize_tool_arguments_with_empty_dict() -> None:
    """Test normalizing empty dict arguments."""
    # Test case 5: Empty dict
    empty_dict = "{}"
    result = normalize_tool_arguments(empty_dict)
    assert result == "{}"


def test_normalize_tool_arguments_with_already_valid_json() -> None:
    """Test that already valid JSON is returned as-is."""
    # Test case 6: Already valid JSON
    valid_json = '{"key": "value", "number": 42, "bool": true}'
    result = normalize_tool_arguments(valid_json)
    assert result == '{"key": "value", "number": 42, "bool": true}'


def test_normalize_tool_arguments_with_special_characters() -> None:
    """Test normalizing arguments with special characters."""
    # Test case 7: Special characters and escaped strings
    special_chars = '{"message": "Hello \\"world\\"!", "path": "C:\\\\\\\\Users"}'
    result = normalize_tool_arguments(special_chars)

    parsed = json.loads(result)
    assert isinstance(parsed, dict)
    assert "message" in parsed


def test_normalize_tool_arguments_with_numbers_and_booleans() -> None:
    """Test normalizing arguments with various data types."""
    # Test case 8: Mixed data types
    mixed_types = '{"temp": 25.5, "enabled": true, "count": 10, "data": null}'
    result = normalize_tool_arguments(mixed_types)

    parsed = json.loads(result)
    assert parsed["temp"] == 25.5
    assert parsed["enabled"] is True
    assert parsed["count"] == 10
    assert parsed["data"] is None
