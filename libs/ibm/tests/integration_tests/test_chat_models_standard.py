import os
from typing import Literal

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests.chat_models import ChatModelIntegrationTests

from langchain_ibm import ChatWatsonx

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"

MODEL_ID = "ibm/granite-3-3-8b-instruct"
MODEL_ID_2 = "meta-llama/llama-3-3-70b-instruct"
MODEL_ID_IMAGE = "meta-llama/llama-3-2-90b-vision-instruct"
MODEL_ID_DOUBLE_MSG_CONV = "meta-llama/llama-3-2-3b-instruct"


class TestChatWatsonxStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
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
    ) -> dict[
        Literal["invoke", "stream"],
        list[
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

    @property
    def has_structured_output(self) -> bool:
        # Required until there is no 'structured_output_format' in the output.
        return False

    @pytest.mark.xfail(reason="Supported for model 2.")
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        model.watsonx_model._inference.model_id = MODEL_ID_2  # type: ignore[attr-defined]
        super().test_tool_message_histories_list_content(model, my_adder_tool)
        model.watsonx_model._inference.model_id = MODEL_ID  # type: ignore[attr-defined]

    @pytest.mark.xfail(reason="Supported for vision model 2.")
    def test_message_with_name(self, model: BaseChatModel) -> None:
        model.watsonx_model._inference.model_id = MODEL_ID_2  # type: ignore[attr-defined]
        super().test_message_with_name(model)
        model.watsonx_model._inference.model_id = MODEL_ID  # type: ignore[attr-defined]

    @pytest.mark.xfail(reason="Supported for vision model.")
    def test_image_inputs(self, model: BaseChatModel) -> None:
        model.watsonx_model._inference.model_id = MODEL_ID_IMAGE  # type: ignore[attr-defined]
        super().test_image_inputs(model)
        model.watsonx_model._inference.model_id = MODEL_ID  # type: ignore[attr-defined]

    @pytest.mark.xfail(reason="Not implemented tool_choice as `any`.")
    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_structured_few_shot_examples(model, my_adder_tool)

    @pytest.mark.xfail(
        reason="Base model does not support double messages conversation."
    )
    def test_double_messages_conversation(self, model: BaseChatModel) -> None:
        model.watsonx_model._inference.model_id = MODEL_ID_DOUBLE_MSG_CONV  # type: ignore[attr-defined]
        super().test_double_messages_conversation(model)
        model.watsonx_model._inference.model_id = MODEL_ID  # type: ignore[attr-defined]
