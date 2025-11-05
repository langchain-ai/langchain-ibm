import os

import pytest
from ibm_watsonx_ai import APIClient, Credentials  # type: ignore[import-untyped]
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from langchain_ibm import ChatWatsonx

URL = "https://us-south.ml.cloud.ibm.com"
WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

MODEL = "mistralai/mistral-large"


@pytest.mark.skip("Until not released on us-south, should be skipped.")
class TestChatModelsGateway:
    prompt = "Hello, How are you!"

    messages = (
        ("user", "You are a helpful assistant that translates English to French."),
        (
            "human",
            "Translate this sentence from English to French. I love programming.",
        ),
    )

    system_message = SystemMessage(
        content="You are a helpful assistant which telling "
        "short-info about provided topic."
    )
    human_message = HumanMessage(content="horse")

    def test_chat_model_gateway_init_credentials(self) -> None:
        chat = ChatWatsonx(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        assert isinstance(chat, ChatWatsonx)

        response = chat.invoke(self.prompt)

        assert response
        assert response.content

    def test_chat_model_gateway_init_client(self) -> None:
        credentials = Credentials(url=URL, api_key=WX_APIKEY)
        api_client = APIClient(credentials=credentials, project_id=WX_PROJECT_ID)

        chat = ChatWatsonx(model=MODEL, watsonx_client=api_client)
        assert isinstance(chat, ChatWatsonx)

        response = chat.invoke(self.prompt)

        assert response
        assert response.content

    def test_chat_model_gateway_invoke(self) -> None:
        chat = ChatWatsonx(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        response = chat.invoke(self.prompt)
        assert response
        assert response.content

        response = chat.invoke(self.messages)
        assert response
        assert response.content

    async def test_chat_model_gateway_ainvoke(self) -> None:
        chat = ChatWatsonx(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        response = await chat.ainvoke(self.prompt)
        assert response
        assert response.content

        response = await chat.ainvoke(self.messages)
        assert response
        assert response.content

    def test_chat_model_gateway_generate(self) -> None:
        chat = ChatWatsonx(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        response = chat.generate(messages=[[self.system_message], [self.human_message]])
        assert response
        for generation in response.generations:
            assert generation[0].text

    async def test_chat_model_gateway_agenerate(self) -> None:
        chat = ChatWatsonx(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        response = await chat.agenerate([[self.system_message], [self.human_message]])
        assert response
        for generation in response.generations:
            assert generation[0].text

    def test_chat_model_gateway_stream(self) -> None:
        chat = ChatWatsonx(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        response = chat.stream(self.prompt)
        for chunk in response:
            assert isinstance(chunk.content, str)

        response = chat.stream(self.messages)
        for chunk in response:
            assert isinstance(chunk.content, str)

    async def test_chat_model_gateway_astream(self) -> None:
        chat = ChatWatsonx(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        response = chat.astream(self.prompt)
        async for chunk in response:
            assert isinstance(chunk.content, str)

        response = chat.astream(self.messages)
        async for chunk in response:
            assert isinstance(chunk.content, str)
