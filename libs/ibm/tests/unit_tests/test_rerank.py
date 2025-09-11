"""Test WatsonxLLM API wrapper."""

import os

import pytest
from ibm_watsonx_ai.wml_client_error import WMLClientError  # type: ignore

from langchain_ibm import WatsonxRerank

os.environ.pop("WATSONX_APIKEY", None)
os.environ.pop("WATSONX_PROJECT_ID", None)

MODEL_ID = "sample_rerank_model"


def test_initialize_watsonxllm_bad_path_without_url() -> None:
    try:
        WatsonxRerank(
            model_id=MODEL_ID,
        )
    except ValueError as e:
        assert "url" in e.__str__()
        assert "WATSONX_URL" in e.__str__()


def test_initialize_watsonxllm_cloud_bad_path() -> None:
    try:
        WatsonxRerank(model_id=MODEL_ID, url="https://us-south.ml.cloud.ibm.com")  # type: ignore[arg-type]
    except ValueError as e:
        assert "apikey" in e.__str__() and "token" in e.__str__()
        assert "WATSONX_APIKEY" in e.__str__() and "WATSONX_TOKEN" in e.__str__()


def test_initialize_watsonxllm_cpd_bad_path_without_all() -> None:
    try:
        WatsonxRerank(
            model_id=MODEL_ID,
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
        WatsonxRerank(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            password="test_password",  # type: ignore[arg-type]
        )
    except ValueError as e:
        assert "username" in e.__str__()
        assert "WATSONX_USERNAME" in e.__str__()


def test_initialize_watsonxllm_cpd_bad_path_apikey_without_username() -> None:
    try:
        WatsonxRerank(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
        )
    except ValueError as e:
        assert "username" in e.__str__()
        assert "WATSONX_USERNAME" in e.__str__()


def test_initialize_watsonxllm_cpd_deprecation_warning_with_instance_id() -> None:
    with pytest.warns(
        DeprecationWarning, match="The `instance_id` parameter is deprecated"
    ):
        with pytest.raises(WMLClientError):
            WatsonxRerank(
                model_id=MODEL_ID,
                url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
                apikey="test_apikey",  # type: ignore[arg-type]
                username="test_user",  # type: ignore[arg-type]
                instance_id="openshift",  # type: ignore[arg-type]
            )
