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
        WatsonxToolkit(url="https://us-south.ml.cloudXXX.ibm.com")
    except ValueError as e:
        assert "url" in e.__str__()


def test_initialize_watsonx_toolkit_cloud_bad_api_key() -> None:
    try:
        WatsonxToolkit(url="https://us-south.ml.cloud.ibm.com")
    except ValueError as e:
        assert "api_key" in e.__str__() and "token" in e.__str__()
        assert "WATSONX_API_KEY" in e.__str__() and "WATSONX_TOKEN" in e.__str__()


def test_initialize_watsonx_toolkit_with_deprecated_apikey() -> None:
    with (
        pytest.warns(
            DeprecationWarning,
            match="'apikey' parameter is deprecated; use 'api_key' instead.",
        ),
        pytest.raises(WMLClientError),
    ):
        WatsonxToolkit(
            url="https://us-south.ml.cloud.ibm.com",
            apikey="test_apikey",
        )


def test_initialize_watsonx_toolkit_with_api_key_and_apikey() -> None:
    with (
        pytest.warns(
            UserWarning,
            match="Both 'api_key' and deprecated 'apikey' were provided; "
            "'api_key' takes precedence.",
        ),
        pytest.raises(WMLClientError),
    ):
        WatsonxToolkit(
            url="https://us-south.ml.cloud.ibm.com",
            apikey="fake_apikey",
            api_key="fake_api_key",
        )


def test_initialize_watsonx_toolkit_cpd_bad_path_without_all() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxToolkit(
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
        )
    assert (
        "api_key" in str(e.value)
        and "password" in str(e.value)
        and "token" in str(e.value)
    )
    assert (
        "WATSONX_API_KEY" in str(e.value)
        and "WATSONX_PASSWORD" in str(e.value)
        and "WATSONX_TOKEN" in str(e.value)
    )


def test_initialize_watsonx_toolkit_cpd_bad_path_password_without_username() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxToolkit(
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            password="test_password",
        )
    assert "username" in str(e.value)
    assert "WATSONX_USERNAME" in str(e.value)


def test_initialize_watsonx_toolkit_cpd_bad_path_apikey_without_username() -> None:
    with pytest.raises(ValueError) as e:
        WatsonxToolkit(
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            apikey="test_apikey",
        )

    assert "username" in str(e.value)
    assert "WATSONX_USERNAME" in str(e.value)


def test_initialize_watsonx_embeddings_cpd_deprecation_warning_with_instance_id() -> (
    None
):
    with (
        pytest.warns(
            DeprecationWarning, match="The `instance_id` parameter is deprecated"
        ),
        pytest.raises(WMLClientError),
    ):
        WatsonxToolkit(
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            apikey="test_apikey",
            username="test_user",
            instance_id="openshift",
        )
