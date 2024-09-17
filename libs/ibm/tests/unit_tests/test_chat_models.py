"""Test ChatWatsonx API wrapper."""

import os

from langchain_ibm import ChatWatsonx

os.environ.pop("WATSONX_APIKEY", None)
os.environ.pop("WATSONX_PROJECT_ID", None)

MODEL_ID = "mistralai/mixtral-8x7b-instruct-v01"


def test_initialize_chat_watsonx_bad_path_without_url() -> None:
    try:
        ChatWatsonx(
            model_id=MODEL_ID,
        )
    except ValueError as e:
        assert "url" in e.__str__()
        assert "WATSONX_URL" in e.__str__()


def test_initialize_chat_watsonx_cloud_bad_path() -> None:
    try:
        ChatWatsonx(model_id=MODEL_ID, url="https://us-south.ml.cloud.ibm.com")  # type: ignore[arg-type]
    except ValueError as e:
        assert "apikey" in e.__str__()
        assert "WATSONX_APIKEY" in e.__str__()


def test_initialize_chat_watsonx_cpd_bad_path_without_all() -> None:
    try:
        ChatWatsonx(
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


def test_initialize_chat_watsonx_cpd_bad_path_password_without_username() -> None:
    try:
        ChatWatsonx(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            password="test_password",  # type: ignore[arg-type]
        )
    except ValueError as e:
        assert "username" in e.__str__()
        assert "WATSONX_USERNAME" in e.__str__()


def test_initialize_chat_watsonx_cpd_bad_path_apikey_without_username() -> None:
    try:
        ChatWatsonx(
            model_id=MODEL_ID,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
        )
    except ValueError as e:
        assert "username" in e.__str__()
        assert "WATSONX_USERNAME" in e.__str__()


def test_initialize_chat_watsonx_cpd_bad_path_without_instance_id() -> None:
    try:
        ChatWatsonx(
            model_id="google/flan-ul2",
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
            username="test_user",  # type: ignore[arg-type]
        )
    except ValueError as e:
        assert "instance_id" in e.__str__()
        assert "WATSONX_INSTANCE_ID" in e.__str__()
