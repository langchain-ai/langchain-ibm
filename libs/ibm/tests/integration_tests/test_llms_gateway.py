import os

import pytest
from ibm_watsonx_ai import APIClient, Credentials  # type: ignore

from langchain_ibm import WatsonxLLM

URL = "https://us-south.ml.cloud.ibm.com"
WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

MODEL = "ibm/granite-3-8b-instruct"


@pytest.mark.skip("Until not released on us-south, should be skipped.")
class TestLLMGateway:
    prompt = "Hello, How are you!"

    def test_llm_model_gateway_init_credentials(self) -> None:
        chat = WatsonxLLM(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        assert isinstance(chat, WatsonxLLM)

        response = chat.invoke(self.prompt)

        assert response
        assert isinstance(response, str)

    def test_llm_model_gateway_init_client(self) -> None:
        credentials = Credentials(url=URL, api_key=WX_APIKEY)
        api_client = APIClient(credentials=credentials, project_id=WX_PROJECT_ID)

        chat = WatsonxLLM(model=MODEL, watsonx_client=api_client)
        assert isinstance(chat, WatsonxLLM)

        response = chat.invoke(self.prompt)

        assert response
        assert isinstance(response, str)

    def test_llm_model_gateway_invoke(self) -> None:
        chat = WatsonxLLM(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        response = chat.invoke(self.prompt)
        assert response
        assert isinstance(response, str)

    async def test_llm_model_gateway_ainvoke(self) -> None:
        chat = WatsonxLLM(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        response = await chat.ainvoke(self.prompt)
        assert response
        assert isinstance(response, str)

    def test_llm_model_gateway_generate(self) -> None:
        chat = WatsonxLLM(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        response = chat.generate(["What color sunflower is?"])
        assert response
        for generation in response.generations:
            assert generation[0].text

    async def test_llm_model_gateway_agenerate(self) -> None:
        chat = WatsonxLLM(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        response = await chat.agenerate(["What color sunflower is?"])
        assert response
        for generation in response.generations:
            assert generation[0].text

    def test_chat_model_gateway_stream(self) -> None:
        chat = WatsonxLLM(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        response = chat.stream(self.prompt)
        for chunk in response:
            assert isinstance(chunk, str)

    async def test_chat_model_gateway_astream(self) -> None:
        chat = WatsonxLLM(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        response = chat.astream(self.prompt)
        async for chunk in response:
            assert isinstance(chunk, str)
