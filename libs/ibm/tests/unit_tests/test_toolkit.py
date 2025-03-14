"""Test IBM watsonx.ai Toolkit API wrapper."""

import os

from langchain_ibm import WatsonxToolkit

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
