import os
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_ibm import ChatWatsonx

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"

MODEL_ID = "mistralai/mistral-large"


class TestTogetherStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatWatsonx

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_id": MODEL_ID,
            "url": URL,
            "apikey": WX_APIKEY,
            "project_id": WX_PROJECT_ID,
        }

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_structured_output(self, model: BaseChatModel) -> None:
        super().test_structured_output(model)

    @pytest.mark.xfail(reason="Not yet implemented.")
    async def test_structured_output_async(self, model: BaseChatModel) -> None:  # type: ignore[override]
        await super().test_structured_output_async(model)

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_structured_few_shot_examples(self, model: BaseChatModel) -> None:
        super().test_structured_few_shot_examples(model)
