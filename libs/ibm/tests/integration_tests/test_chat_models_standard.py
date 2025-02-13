import os
from typing import Dict, List, Literal, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_ibm import ChatWatsonx

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"

MODEL_ID = "mistralai/mistral-large"
MODEL_ID_IMAGE = "meta-llama/llama-3-2-11b-vision-instruct"


class TestChatWatsonxStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatWatsonx

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def returns_usage_metadata(self) -> bool:
        return True

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supported_usage_metadata_details(
        self,
    ) -> Dict[
        Literal["invoke", "stream"],
        List[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        return {
            "invoke": [],
            "stream": [],
        }

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_id": MODEL_ID,
            "url": URL,
            "apikey": WX_APIKEY,
            "project_id": WX_PROJECT_ID,
            "temperature": 0,
        }

    @pytest.mark.xfail(reason="Supported for vision model.")
    def test_image_inputs(self, model: BaseChatModel) -> None:
        model.watsonx_model._inference.model_id = MODEL_ID_IMAGE  # type: ignore[attr-defined]
        super().test_image_inputs(model)
        model.watsonx_model._inference.model_id = MODEL_ID  # type: ignore[attr-defined]

    @pytest.mark.xfail(reason="Not implemented tool_choice as `any`.")
    def test_structured_few_shot_examples(self, model: BaseChatModel) -> None:
        super().test_structured_few_shot_examples(model)
