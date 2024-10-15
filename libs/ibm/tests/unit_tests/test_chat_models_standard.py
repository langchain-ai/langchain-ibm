import os
from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_ibm import ChatWatsonx

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")


class TestGeminiAIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatWatsonx

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_id": "mistralai/mistral-large",
            "url": "https://us-south.ml.cloud.ibm.com",
            "project_id": WX_PROJECT_ID,
        }
