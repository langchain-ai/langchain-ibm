"""Test WatsonxLLM API wrapper."""

import os
from unittest.mock import Mock

import pytest
from ibm_watsonx_ai import APIClient  # type: ignore
from ibm_watsonx_ai.foundation_models import Model, ModelInference  # type: ignore
from ibm_watsonx_ai.gateway import Gateway  # type: ignore
from requests.exceptions import ConnectionError

from langchain_ibm import WatsonxLLM

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

model_mock = Mock(spec=Model)
model_mock._client = api_client_mock
model_mock.model_id = "fake_model_id"
model_mock.params = {"temperature": 1}


def test_initialize_watsonxllm_bad_path_without_url() -> None:
    try:
        WatsonxLLM(
            model_id="google/flan-ul2",
        )
    except ValueError as e:
        assert "url" in e.__str__()
        assert "WATSONX_URL" in e.__str__()


def test_initialize_watsonxllm_cloud_bad_path() -> None:
    try:
        WatsonxLLM(model_id="google/flan-ul2", url="https://us-south.ml.cloud.ibm.com")  # type: ignore[arg-type]
    except ValueError as e:
        assert "apikey" in e.__str__() and "token" in e.__str__()
        assert "WATSONX_APIKEY" in e.__str__() and "WATSONX_TOKEN" in e.__str__()


def test_initialize_watsonxllm_cpd_bad_path_without_all() -> None:
    try:
        WatsonxLLM(
            model_id="google/flan-ul2",
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
        )
    except ValueError as e:
        assert (
            "apikey" in e.__str__()
            and "password" in e.__str__()
            and "token" in e.__str__()
        )
        assert (
            "WATSONX_APIKEY" in e.__str__()
            and "WATSONX_PASSWORD" in e.__str__()
            and "WATSONX_TOKEN" in e.__str__()
        )


def test_initialize_watsonxllm_cpd_bad_path_password_without_username() -> None:
    try:
        WatsonxLLM(
            model_id="google/flan-ul2",
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            password="test_password",  # type: ignore[arg-type]
        )
    except ValueError as e:
        assert "username" in e.__str__()
        assert "WATSONX_USERNAME" in e.__str__()


def test_initialize_watsonxllm_cpd_bad_path_apikey_without_username() -> None:
    try:
        WatsonxLLM(
            model_id="google/flan-ul2",
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
        )
    except ValueError as e:
        assert "username" in e.__str__()
        assert "WATSONX_USERNAME" in e.__str__()


def test_initialize_watsonxllm_cpd_deprecation_warning_with_instance_id() -> None:
    with pytest.warns(DeprecationWarning) as w:
        with pytest.raises(ConnectionError):
            WatsonxLLM(
                model_id="google/flan-ul2",
                url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
                apikey="test_apikey",  # type: ignore[arg-type]
                username="test_user",  # type: ignore[arg-type]
                instance_id="openshift",  # type: ignore[arg-type]
            )
    assert "The `instance_id` parameter is deprecated" in str(w[-1].message)


def test_initialize_watsonxllm_with_two_exclusive_parameters() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxLLM(
            model_id=MODEL_ID,
            model=MODEL_ID,
            url="https://us-south.ml.cloud.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
        )

    assert (
        "The parameters 'model', 'model_id' and 'deployment_id' are mutually exclusive."
        " Please specify exactly one of these parameters when initializing WatsonxLLM."
        in str(e.value)
    )


def test_initialize_watsonxllm_with_three_exclusive_parameters() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxLLM(
            model_id=MODEL_ID,
            model=MODEL_ID,
            deployment_id="test_deployment_id",
            url="https://us-south.ml.cloud.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
        )

    assert (
        "The parameters 'model', 'model_id' and 'deployment_id' are mutually exclusive."
        " Please specify exactly one of these parameters when initializing WatsonxLLM."
        in str(e.value)
    )


def test_initialize_watsonxllm_with_api_client_only() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxLLM(watsonx_client=api_client_mock)
    assert (
        "The parameters 'model', 'model_id' and 'deployment_id' are mutually exclusive."
        " Please specify exactly one of these parameters when initializing WatsonxLLM."
        in str(e.value)
    )


def test_initialize_watsonxllm_with_watsonx_model_gateway() -> None:
    with pytest.raises(NotImplementedError) as e:
        WatsonxLLM(watsonx_model_gateway=gateway_mock)
    assert (
        "Passing the 'watsonx_model_gateway' parameter to the WatsonxLLM "
        "constructor is not supported yet." in str(e.value)
    )


def test_initialize_watsonxllm_without_any_params() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxLLM()
    assert (
        "The parameters 'model', 'model_id' and 'deployment_id' are mutually exclusive."
        " Please specify exactly one of these parameters when initializing WatsonxLLM."
        in str(e.value)
    )


def test_initialize_watsonxllm_with_model_inference_only() -> None:
    chat = WatsonxLLM(watsonx_model=model_inference_mock)

    assert isinstance(chat, WatsonxLLM)


def test_initialize_watsonxllm_with_model_only() -> None:
    chat = WatsonxLLM(watsonx_model=model_mock)

    assert isinstance(chat, WatsonxLLM)
