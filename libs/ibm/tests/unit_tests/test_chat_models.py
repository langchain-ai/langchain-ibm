"""Test ChatWatsonx API wrapper."""

import os
from typing import Any
from unittest.mock import Mock

import pytest
from ibm_watsonx_ai import APIClient  # type: ignore
from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
from ibm_watsonx_ai.gateway import Gateway  # type: ignore

from langchain_ibm import ChatWatsonx

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
        ChatWatsonx(model_id=MODEL_ID, url="https://us-south.ml.cloud.ibm.com")  # type: ignore[arg-type]

    assert "apikey" in str(e.value) and "token" in str(e.value)
    assert "WATSONX_APIKEY" in str(e.value) and "WATSONX_TOKEN" in str(e.value)


def test_initialize_chat_watsonx_cpd_bad_path_without_all() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
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


def test_initialize_chat_watsonx_cpd_bad_path_password_without_username() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            password="test_password",  # type: ignore[arg-type]
        )
    assert "username" in str(e.value)
    assert "WATSONX_USERNAME" in str(e.value)


def test_initialize_chat_watsonx_cpd_bad_path_apikey_without_username() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
        )

    assert "username" in str(e.value)
    assert "WATSONX_USERNAME" in str(e.value)


def test_initialize_chat_watsonx_cpd_bad_path_without_instance_id() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(
            model_id="google/flan-ul2",
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
            username="test_user",  # type: ignore[arg-type]
        )
    assert "instance_id" in str(e.value)
    assert "WATSONX_INSTANCE_ID" in str(e.value)


def test_initialize_chat_watsonx_with_two_exclusive_parameters() -> None:
    with pytest.raises(ValueError) as e:
        ChatWatsonx(
            model_id=MODEL_ID,
            model=MODEL_ID,
            url="https://us-south.ml.cloud.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
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
            url="https://us-south.ml.cloud.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
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


def test_initialize_chat_watsonx_with_model_inferencey_only() -> None:
    chat = ChatWatsonx(watsonx_model=model_inference_mock)

    assert isinstance(chat, ChatWatsonx)


def test_initialize_chat_watsonx_with_all_supported_params(mocker: Any) -> None:
    # All params values are taken from
    # ibm_watsonx_ai.foundation_models.schema.TextChatParameters.get_sample_params()

    from ibm_watsonx_ai.foundation_models.schema import (  # type: ignore[import-not-found, import-untyped]
        TextChatParameters,
    )

    TOP_P = 0.8

    def mock_modelinference_chat(*args: Any, **kwargs: Any) -> dict:
        """Mock ModelInference.chat method"""

        assert kwargs.get("params", None) == (
            TextChatParameters.get_sample_params()
            | dict(
                logit_bias={"1003": -100, "1004": -100}, seed=41, stop=["this", "the"]
            )
            | dict(top_p=TOP_P)
        )
        # logit_bias, seed and stop available in sdk since 1.2.7
        return {"id": "123", "choices": [{"message": dict(content="Hi", role="ai")}]}

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
        url="https://us-south.ml.cloud.ibm.com",  # type: ignore[arg-type]
        apikey="test_apikey",  # type: ignore[arg-type]
        frequency_penalty=0.5,
        logprobs=True,
        top_logprobs=3,
        presence_penalty=0.3,
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=100,
        time_limit=600000,
        top_p=0.9,
        n=1,
        logit_bias={"1003": -100, "1004": -100},
        seed=41,
        stop=["this", "the"],
    )

    # change only top_n
    chat.invoke("Hello", top_p=TOP_P)
