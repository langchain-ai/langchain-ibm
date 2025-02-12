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
        assert "apikey" in e.__str__() and "token" in e.__str__()
        assert "WATSONX_APIKEY" in e.__str__() and "WATSONX_TOKEN" in e.__str__()


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


def test_initialize_chat_watsonx_with_all_supported_params(mocker) -> None:
    # All params values are taken from ibm_watsonx_ai.foundation_models.schema.TextChatParameters.get_sample_params().
    from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
    from langchain_core.messages import ChatMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    TOP_P = 0.8

    def mock_modelinference_chat(*args, **kwargs) -> dict:
        assert kwargs.get("params", None) == (
            TextChatParameters.get_sample_params()
            | dict(
                logit_bias={"1003": -100, "1004": -100}, seed=41, stop=["this", "the"]
            )
            | dict(top_p=TOP_P)
        )
        # logit_bias, seed and stop available in sdk since 1.2.7

    with mocker.patch(
        "ibm_watsonx_ai.foundation_models.ModelInference.__init__", return_value=None
    ) as mock, mocker.patch(
        "ibm_watsonx_ai.foundation_models.ModelInference.chat",
        side_effect=mock_modelinference_chat,
    ), mocker.patch.object(
        ChatWatsonx,
        "_create_chat_result",
        return_value=ChatResult(
            generations=[ChatGeneration(message=ChatMessage(content="Hi", role="ai"))],
            llm_output={},
        ),
    ):
        chat = ChatWatsonx(
            model_id="google/flan-ul2",
            url="https://us-south.ml.cloud.ibm.com",  # type: ignore[arg-type]
            apikey="test_apikey",  # type: ignore[arg-type]
            frequency_penalty=0.5,
            logprobs=True,
            top_logprobs=3,
            presence_penalty=0.3,
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=100,
            time_limit=600000,
            top_p=0.9,
            n=1,
            logit_bias={"1003": -100, "1004": -100},
            seed=41,
            stop=["this", "the"],
        )

        # change only top_n
        chat.invoke("Hello", top_p=TOP_P)
