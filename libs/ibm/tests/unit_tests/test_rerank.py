"""Test WatsonxRerank API wrapper."""

import os
from typing import Any

import pytest
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
)

from langchain_ibm import WatsonxRerank

os.environ.pop("WATSONX_APIKEY", None)
os.environ.pop("WATSONX_PROJECT_ID", None)
os.environ.pop("WATSONX_SPACE_ID", None)

MODEL_ID = "sample_rerank_model"


def test_initialize_watsonx_rerank_without_url() -> None:
    pattern = r"(?=.*url)(?=.*WATSONX_URL)"
    with pytest.raises(ValueError, match=pattern):
        WatsonxRerank(
            model_id=MODEL_ID,
        )


def test_initialize_watsonx_rerank_cloud_only_url() -> None:
    pattern = (
        r"(?=.*api_key)(?=.*token)"
        r"(?=.*WATSONX_API_KEY)(?=.*WATSONX_TOKEN)"
    )
    with pytest.raises(ValueError, match=pattern):
        WatsonxRerank(model_id=MODEL_ID, url="https://us-south.ml.cloud.ibm.com")


def test_initialize_watsonx_rerank_with_deprecated_apikey() -> None:
    with (
        pytest.warns(
            DeprecationWarning,
            match="'apikey' parameter is deprecated; use 'api_key' instead.",
        ),
        pytest.raises(WMLClientError),
    ):
        WatsonxRerank(
            model_id=MODEL_ID,
            url="https://us-south.ml.cloud.ibm.com",
            apikey="test_apikey",
        )


def test_initialize_watsonx_rerank_with_api_key_and_apikey() -> None:
    with (
        pytest.warns(
            UserWarning,
            match="Both 'api_key' and deprecated 'apikey' were provided; "
            "'api_key' takes precedence.",
        ),
        pytest.raises(WMLClientError),
    ):
        WatsonxRerank(
            model_id=MODEL_ID,
            url="https://us-south.ml.cloud.ibm.com",
            apikey="fake_apikey",
            api_key="fake_api_key",
        )


def test_initialize_watsonx_rerank_cpd_without_all() -> None:
    pattern = (
        r"(?=.*api_key)(?=.*password)(?=.*token)"
        r"(?=.*WATSONX_API_KEY)(?=.*WATSONX_PASSWORD)(?=.*WATSONX_TOKEN)"
    )
    with pytest.raises(ValueError, match=pattern):
        WatsonxRerank(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
        )


def test_initialize_watsonx_rerank_cpd_only_password() -> None:
    pattern = r"(?=.*username)(?=.*WATSONX_USERNAME)"
    with pytest.raises(ValueError, match=pattern):
        WatsonxRerank(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            password="test_password",  # noqa: S106
        )


def test_initialize_watsonx_rerank_cpd_only_apikey() -> None:
    pattern = r"(?=.*username)(?=.*WATSONX_USERNAME)"
    with pytest.raises(ValueError, match=pattern):
        WatsonxRerank(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            apikey="test_apikey",
        )


def test_initialize_watsonx_rerank_cpd_deprecation_warning_with_instance_id() -> None:
    with (
        pytest.warns(
            DeprecationWarning, match="The `instance_id` parameter is deprecated"
        ),
        pytest.raises(WMLClientError),
    ):
        WatsonxRerank(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            apikey="test_apikey",
            username="test_user",
            instance_id="openshift",
        )


def test_initialize_watsonx_rerank_with_project_id_from_env(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that project_id is loaded from WATSONX_PROJECT_ID environment variable."""
    test_project_id = "test_project_id_from_env"
    monkeypatch.setenv("WATSONX_PROJECT_ID", test_project_id)

    mocker.patch(
        "ibm_watsonx_ai.foundation_models.Rerank.__init__",
        return_value=None,
    )

    embeddings = WatsonxRerank(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        apikey="test_apikey",
    )

    assert embeddings.project_id == test_project_id


def test_initialize_watsonx_rerank_with_space_id_from_env(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that space_id is loaded from WATSONX_SPACE_ID environment variable."""
    test_space_id = "test_space_id_from_env"
    monkeypatch.setenv("WATSONX_SPACE_ID", test_space_id)

    mocker.patch(
        "ibm_watsonx_ai.foundation_models.Rerank.__init__",
        return_value=None,
    )

    embeddings = WatsonxRerank(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        apikey="test_apikey",
    )

    assert embeddings.space_id == test_space_id


def test_initialize_watsonx_rerank_with_both_ids_from_env(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that both project_id and space_id are loaded from environment variables."""
    test_project_id = "test_project_id_from_env"
    test_space_id = "test_space_id_from_env"
    monkeypatch.setenv("WATSONX_PROJECT_ID", test_project_id)
    monkeypatch.setenv("WATSONX_SPACE_ID", test_space_id)

    mocker.patch(
        "ibm_watsonx_ai.foundation_models.Rerank.__init__",
        return_value=None,
    )

    embeddings = WatsonxRerank(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        apikey="test_apikey",
    )

    assert embeddings.project_id == test_project_id
    assert embeddings.space_id == test_space_id


def test_initialize_watsonx_rerank_explicit_project_id_overrides_env(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that explicitly provided project_id overrides environment variable."""
    env_project_id = "env_project_id"
    explicit_project_id = "explicit_project_id"
    monkeypatch.setenv("WATSONX_PROJECT_ID", env_project_id)

    mocker.patch(
        "ibm_watsonx_ai.foundation_models.Rerank.__init__",
        return_value=None,
    )

    embeddings = WatsonxRerank(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        apikey="test_apikey",
        project_id=explicit_project_id,
    )

    assert embeddings.project_id == explicit_project_id


def test_initialize_watsonx_rerank_explicit_space_id_overrides_env(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that explicitly provided space_id overrides environment variable."""
    env_space_id = "env_space_id"
    explicit_space_id = "explicit_space_id"
    monkeypatch.setenv("WATSONX_SPACE_ID", env_space_id)

    mocker.patch(
        "ibm_watsonx_ai.foundation_models.Rerank.__init__",
        return_value=None,
    )

    embeddings = WatsonxRerank(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        apikey="test_apikey",
        space_id=explicit_space_id,
    )

    assert embeddings.space_id == explicit_space_id


def test_initialize_watsonx_rerank_without_project_id_or_space_id_env(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that project_id and space_id default to None when not set."""
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    monkeypatch.delenv("WATSONX_SPACE_ID", raising=False)

    mocker.patch(
        "ibm_watsonx_ai.foundation_models.Rerank.__init__",
        return_value=None,
    )

    embeddings = WatsonxRerank(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        apikey="test_apikey",
    )

    assert embeddings.project_id is None
    assert embeddings.space_id is None
