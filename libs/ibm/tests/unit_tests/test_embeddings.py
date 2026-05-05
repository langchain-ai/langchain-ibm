"""Test WatsonxLLM API wrapper."""

import os
import re
from unittest.mock import Mock

import pytest
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models.embeddings import (
    Embeddings,
)
from ibm_watsonx_ai.gateway import Gateway
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
)

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


def test_initialize_watsonx_embeddings_without_url() -> None:
    pattern = r"(?=.*url)(?=.*WATSONX_URL)"
    with pytest.raises(ValueError, match=pattern):
        WatsonxEmbeddings(
            model_id=MODEL_ID,
        )


def test_initialize_watsonx_embeddings_cloud_only_url() -> None:
    pattern = (
        r"(?=.*api_key)(?=.*token)"
        r"(?=.*WATSONX_API_KEY)(?=.*WATSONX_TOKEN)"
    )
    with pytest.raises(ValueError, match=pattern):
        WatsonxEmbeddings(model_id=MODEL_ID, url="https://us-south.ml.cloud.ibm.com")


def test_initialize_watsonx_embeddings_with_deprecated_apikey() -> None:
    with (
        pytest.warns(
            DeprecationWarning,
            match="'apikey' parameter is deprecated; use 'api_key' instead.",
        ),
        pytest.raises(WMLClientError),
    ):
        WatsonxEmbeddings(
            model_id=MODEL_ID,
            url="https://us-south.ml.cloud.ibm.com",
            apikey="test_apikey",
        )


def test_initialize_watsonx_embeddings_with_api_key_and_apikey() -> None:
    with (
        pytest.warns(
            UserWarning,
            match="Both 'api_key' and deprecated 'apikey' were provided; "
            "'api_key' takes precedence.",
        ),
        pytest.raises(WMLClientError),
    ):
        WatsonxEmbeddings(
            model_id=MODEL_ID,
            url="https://us-south.ml.cloud.ibm.com",
            apikey="fake_apikey",
            api_key="fake_api_key",
        )


def test_initialize_watsonx_embeddings_cpd_without_all() -> None:
    pattern = (
        r"(?=.*api_key)(?=.*password)(?=.*token)"
        r"(?=.*WATSONX_API_KEY)(?=.*WATSONX_PASSWORD)(?=.*WATSONX_TOKEN)"
    )
    with pytest.raises(ValueError, match=pattern):
        WatsonxEmbeddings(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
        )


def test_initialize_watsonx_embeddings_cpd_only_password() -> None:
    pattern = r"(?=.*username)(?=.*WATSONX_USERNAME)"
    with pytest.raises(ValueError, match=pattern):
        WatsonxEmbeddings(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            password="test_password",  # noqa: S106
        )


def test_initialize_watsonx_embeddings_cpd_only_apikey() -> None:
    pattern = r"(?=.*username)(?=.*WATSONX_USERNAME)"
    with pytest.raises(ValueError, match=pattern):
        WatsonxEmbeddings(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            apikey="test_apikey",
        )


def test_initialize_watsonx_embeddings_cpd_deprecation_warning_with_instance_id() -> (
    None
):
    with (
        pytest.warns(
            DeprecationWarning, match="The `instance_id` parameter is deprecated"
        ),
        pytest.raises(WMLClientError),
    ):
        WatsonxEmbeddings(
            model_id="google/flan-ul2",
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            apikey="test_apikey",
            username="test_user",
            instance_id="openshift",
        )


def test_initialize_watsonx_embeddings_with_two_exclusive_parameters() -> None:
    pattern = re.escape(
        "The parameters 'model' and 'model_id' are mutually exclusive. "
        "Please specify exactly one of these parameters when initializing "
        "WatsonxEmbeddings."
    )
    with pytest.raises(ValueError, match=pattern):
        WatsonxEmbeddings(
            model_id=MODEL_ID,
            model=MODEL_ID,
            url="https://us-south.ml.cloud.ibm.com",
            apikey="test_apikey",
        )


def test_initialize_watsonx_embeddings_without_any_params() -> None:
    pattern = re.escape(
        "The parameters 'model' and 'model_id' are mutually exclusive. "
        "Please specify exactly one of these parameters when initializing "
        "WatsonxEmbeddings."
    )
    with pytest.raises(ValueError, match=pattern):
        WatsonxEmbeddings()


def test_initialize_watsonx_embeddings_with_api_client_only() -> None:
    pattern = re.escape(
        "The parameters 'model' and 'model_id' are mutually exclusive. "
        "Please specify exactly one of these parameters when initializing "
        "WatsonxEmbeddings."
    )
    with pytest.raises(ValueError, match=pattern):
        WatsonxEmbeddings(watsonx_client=api_client_mock)


def test_initialize_watsonx_embeddings_with_watsonx_embed_gateway() -> None:
    pattern = re.escape(
        "Passing the 'watsonx_embed_gateway' parameter to the WatsonxEmbeddings "
        "constructor is not supported yet."
    )
    with pytest.raises(NotImplementedError, match=pattern):
        WatsonxEmbeddings(watsonx_embed_gateway=gateway_mock)


def test_initialize_watsonx_embeddings_with_watsonx_embed_only() -> None:
    embeddings = WatsonxEmbeddings(watsonx_embed=embeddings_mock)

    assert isinstance(embeddings, WatsonxEmbeddings)
