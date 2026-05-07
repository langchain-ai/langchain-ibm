"""Test WatsonxEmbeddings API wrapper."""

import os
import re
from typing import Any
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
os.environ.pop("WATSONX_SPACE_ID", None)

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
            project_id="fake_project_id",
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
            project_id="fake_project_id",
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
            project_id="fake_project_id",
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


def test_initialize_watsonx_embeddings_with_project_id_from_env(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that project_id is loaded from WATSONX_PROJECT_ID environment variable."""
    test_project_id = "test_project_id_from_env"
    monkeypatch.setenv("WATSONX_PROJECT_ID", test_project_id)

    mocker.patch(
        "ibm_watsonx_ai.foundation_models.Embeddings.__init__",
        return_value=None,
    )

    embeddings = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        apikey="test_apikey",
    )

    assert embeddings.project_id == test_project_id


def test_initialize_watsonx_embeddings_with_space_id_from_env(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that space_id is loaded from WATSONX_SPACE_ID environment variable."""
    test_space_id = "test_space_id_from_env"
    monkeypatch.setenv("WATSONX_SPACE_ID", test_space_id)

    mocker.patch(
        "ibm_watsonx_ai.foundation_models.Embeddings.__init__",
        return_value=None,
    )

    embeddings = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        apikey="test_apikey",
    )

    assert embeddings.space_id == test_space_id


def test_initialize_watsonx_embeddings_with_both_ids_from_env(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that both project_id and space_id are loaded from environment variables."""
    test_project_id = "test_project_id_from_env"
    test_space_id = "test_space_id_from_env"
    monkeypatch.setenv("WATSONX_PROJECT_ID", test_project_id)
    monkeypatch.setenv("WATSONX_SPACE_ID", test_space_id)

    mocker.patch(
        "ibm_watsonx_ai.foundation_models.Embeddings.__init__",
        return_value=None,
    )

    embeddings = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        apikey="test_apikey",
    )

    assert embeddings.project_id == test_project_id
    assert embeddings.space_id == test_space_id


def test_initialize_watsonx_embeddings_explicit_project_id_overrides_env(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that explicitly provided project_id overrides environment variable."""
    env_project_id = "env_project_id"
    explicit_project_id = "explicit_project_id"
    monkeypatch.setenv("WATSONX_PROJECT_ID", env_project_id)

    mocker.patch(
        "ibm_watsonx_ai.foundation_models.Embeddings.__init__",
        return_value=None,
    )

    embeddings = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        apikey="test_apikey",
        project_id=explicit_project_id,
    )

    assert embeddings.project_id == explicit_project_id


def test_initialize_watsonx_embeddings_explicit_space_id_overrides_env(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that explicitly provided space_id overrides environment variable."""
    env_space_id = "env_space_id"
    explicit_space_id = "explicit_space_id"
    monkeypatch.setenv("WATSONX_SPACE_ID", env_space_id)

    mocker.patch(
        "ibm_watsonx_ai.foundation_models.Embeddings.__init__",
        return_value=None,
    )

    embeddings = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        apikey="test_apikey",
        space_id=explicit_space_id,
    )

    assert embeddings.space_id == explicit_space_id


def test_initialize_watsonx_embeddings_model_id_without_project_or_space_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that ValueError is raised when model_id is used without scope."""
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    monkeypatch.delenv("WATSONX_SPACE_ID", raising=False)

    error_pattern = re.escape(
        "When using 'model_id', you must provide either 'project_id' "
        "or 'space_id'. These can be passed as parameters to WatsonxEmbeddings"
        " or set as environment variables 'WATSONX_PROJECT_ID' or "
        "'WATSONX_SPACE_ID'."
    )

    with pytest.raises(ValueError, match=error_pattern):
        WatsonxEmbeddings(
            model_id=MODEL_ID,
            url="https://us-south.ml.cloud.ibm.com",
            apikey="test_apikey",
        )
