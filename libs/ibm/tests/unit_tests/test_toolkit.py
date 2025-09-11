"""Test IBM watsonx.ai Toolkit API wrapper."""

import os

import pytest
from ibm_watsonx_ai.wml_client_error import WMLClientError  # type: ignore

from langchain_ibm.agent_toolkits.utility import WatsonxToolkit

os.environ.pop("WATSONX_APIKEY", None)
os.environ.pop("WATSONX_PROJECT_ID", None)


def test_initialize_watsonx_toolkit_bad_path_without_url() -> None:
    try:
        WatsonxToolkit()
    except ValueError as e:
        assert "url" in e.__str__()
        assert "WATSONX_URL" in e.__str__()


def test_initialize_watsonx_toolkit_cloud_bad_path() -> None:
    try:
        WatsonxToolkit(url="https://us-south.ml.cloudXXX.ibm.com")  # type: ignore[arg-type]
    except ValueError as e:
        assert "url" in e.__str__()


def test_initialize_watsonx_toolkit_cloud_bad_api_key() -> None:
    try:
        WatsonxToolkit(url="https://us-south.ml.cloud.ibm.com")  # type: ignore[arg-type]
    except ValueError as e:
        assert "apikey" in e.__str__() and "token" in e.__str__()
        assert "WATSONX_APIKEY" in e.__str__() and "WATSONX_TOKEN" in e.__str__()


def test_initialize_watsonx_toolkit_cpd_bad_path_without_all() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxToolkit(
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


def test_initialize_watsonx_toolkit_cpd_bad_path_password_without_username() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxToolkit(
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            password="test_password",  # type: ignore[arg-type]
        )
    assert "username" in str(e.value)
    assert "WATSONX_USERNAME" in str(e.value)


def test_initialize_watsonx_toolkit_cpd_bad_path_apikey_without_username() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxToolkit(
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
        )

    assert "username" in str(e.value)
    assert "WATSONX_USERNAME" in str(e.value)


def test_initialize_watsonx_embeddings_cpd_deprecation_warning_with_instance_id() -> (
    None
):
    with pytest.warns(
        DeprecationWarning, match="The `instance_id` parameter is deprecated"
    ):
        with pytest.raises(WMLClientError):
            WatsonxToolkit(
                url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
                apikey="test_apikey",  # type: ignore[arg-type]
                username="test_user",  # type: ignore[arg-type]
                instance_id="openshift",  # type: ignore[arg-type]
            )
