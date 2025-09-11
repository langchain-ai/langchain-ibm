"""Test WatsonxLLM API wrapper."""

import os
from unittest.mock import Mock

import pytest
from ibm_watsonx_ai import APIClient  # type: ignore
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings  # type: ignore
from ibm_watsonx_ai.gateway import Gateway  # type: ignore
from ibm_watsonx_ai.wml_client_error import WMLClientError  # type: ignore

from langchain_ibm import WatsonxEmbeddings

os.environ.pop("WATSONX_APIKEY", None)
os.environ.pop("WATSONX_PROJECT_ID", None)

MODEL_ID = "ibm/granite-embedding-107m-multilingual"

api_client_mock = Mock(spec=APIClient)
api_client_mock.default_space_id = None
api_client_mock.default_project_id = "fake_project_id"

gateway_mock = Mock(spec=Gateway)
gateway_mock._client = api_client_mock

embeddings_mock = Mock(spec=Embeddings)
embeddings_mock._client = api_client_mock
embeddings_mock.model_id = "fake_model_id"
embeddings_mock.params = {"temperature": 1}


def test_initialize_watsonx_embeddings_bad_path_without_url() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxEmbeddings(
            model_id=MODEL_ID,
        )
    assert "url" in str(e.value)
    assert "WATSONX_URL" in str(e.value)


def test_initialize_watsonx_embeddings_cloud_bad_path() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxEmbeddings(model_id=MODEL_ID, url="https://us-south.ml.cloud.ibm.com")  # type: ignore[arg-type]

    assert "apikey" in str(e.value) and "token" in str(e.value)
    assert "WATSONX_APIKEY" in str(e.value) and "WATSONX_TOKEN" in str(e.value)


def test_initialize_watsonx_embeddings_cpd_bad_path_without_all() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxEmbeddings(
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


def test_initialize_watsonx_embeddings_cpd_bad_path_password_without_username() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxEmbeddings(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            password="test_password",  # type: ignore[arg-type]
        )
    assert "username" in str(e.value)
    assert "WATSONX_USERNAME" in str(e.value)


def test_initialize_watsonx_embeddings_cpd_bad_path_apikey_without_username() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxEmbeddings(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
        )
    assert "username" in str(e.value)
    assert "WATSONX_USERNAME" in str(e.value)


def test_initialize_watsonx_embeddings_cpd_deprecation_warning_with_instance_id() -> (
    None
):
    with pytest.warns(DeprecationWarning) as w:
        with pytest.raises(WMLClientError):
            WatsonxEmbeddings(
                model_id="google/flan-ul2",
                url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
                apikey="test_apikey",  # type: ignore[arg-type]
                username="test_user",  # type: ignore[arg-type]
                instance_id="openshift",  # type: ignore[arg-type]
            )
    assert "The `instance_id` parameter is deprecated" in str(w[-1].message)


def test_initialize_watsonx_embeddings_with_two_exclusive_parameters() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxEmbeddings(
            model_id=MODEL_ID,
            model=MODEL_ID,
            url="https://us-south.ml.cloud.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
        )

    assert (
        "The parameters 'model' and 'model_id' are mutually exclusive. "
        "Please specify exactly one of these parameters when initializing "
        "WatsonxEmbeddings." in str(e.value)
    )


def test_initialize_watsonx_embeddings_without_any_params() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxEmbeddings()
    assert (
        "The parameters 'model' and 'model_id' are mutually exclusive. "
        "Please specify exactly one of these parameters when initializing "
        "WatsonxEmbeddings." in str(e.value)
    )


def test_initialize_watsonx_embeddings_with_api_client_only() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxEmbeddings(watsonx_client=api_client_mock)
    assert (
        "The parameters 'model' and 'model_id' are mutually exclusive. "
        "Please specify exactly one of these parameters when initializing "
        "WatsonxEmbeddings." in str(e.value)
    )


def test_initialize_watsonx_embeddings_with_watsonx_embed_gateway() -> None:
    with pytest.raises(NotImplementedError) as e:
        WatsonxEmbeddings(watsonx_embed_gateway=gateway_mock)
    assert (
        "Passing the 'watsonx_embed_gateway' parameter to the WatsonxEmbeddings "
        "constructor is not supported yet." in e.value.__str__()
    )


def test_initialize_watsonx_embeddings_with_watsonx_embed_only() -> None:
    embeddings = WatsonxEmbeddings(watsonx_embed=embeddings_mock)

    assert isinstance(embeddings, WatsonxEmbeddings)
