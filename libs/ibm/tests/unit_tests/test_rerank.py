"""Test WatsonxLLM API wrapper."""

import os

import pytest
from ibm_watsonx_ai.wml_client_error import WMLClientError  # type: ignore

from langchain_ibm import WatsonxRerank

os.environ.pop("WATSONX_APIKEY", None)
os.environ.pop("WATSONX_PROJECT_ID", None)

MODEL_ID = "sample_rerank_model"


def test_initialize_watsonx_rerank_bad_path_without_url() -> None:
    try:
        WatsonxRerank(
            model_id=MODEL_ID,
        )
    except ValueError as e:
        assert "url" in e.__str__()
        assert "WATSONX_URL" in e.__str__()


def test_initialize_watsonx_rerank_cloud_bad_path() -> None:
    try:
        WatsonxRerank(model_id=MODEL_ID, url="https://us-south.ml.cloud.ibm.com")
    except ValueError as e:
        assert "api_key" in e.__str__() and "token" in e.__str__()
        assert "WATSONX_API_KEY" in e.__str__() and "WATSONX_TOKEN" in e.__str__()


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


def test_initialize_watsonx_rerank_cpd_bad_path_without_all() -> None:
    try:
        WatsonxRerank(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
        )
    except ValueError as e:
        assert (
            "api_key" in e.__str__()
            and "password" in e.__str__()
            and "token" in e.__str__()
        )
        assert (
            "WATSONX_API_KEY" in e.__str__()
            and "WATSONX_PASSWORD" in e.__str__()
            and "WATSONX_TOKEN" in e.__str__()
        )


def test_initialize_watsonx_rerank_cpd_bad_path_password_without_username() -> None:
    try:
        WatsonxRerank(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            password="test_password",  # noqa: S106
        )
    except ValueError as e:
        assert "username" in e.__str__()
        assert "WATSONX_USERNAME" in e.__str__()


def test_initialize_watsonx_rerank_cpd_bad_path_apikey_without_username() -> None:
    try:
        WatsonxRerank(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            apikey="test_apikey",
        )
    except ValueError as e:
        assert "username" in e.__str__()
        assert "WATSONX_USERNAME" in e.__str__()


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
