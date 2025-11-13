import os
from typing import Any, Literal

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests.chat_models import ChatModelIntegrationTests
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import-untyped]
from vcr.cassette import Cassette  # type: ignore[import-untyped]

from langchain_ibm import ChatWatsonx

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"

MODEL_ID = "ibm/granite-3-3-8b-instruct"
MODEL_ID_IMAGE = "meta-llama/llama-3-2-90b-vision-instruct"


class TestChatWatsonxStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatWatsonx

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def enable_vcr_tests(self) -> bool:
        return False

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
    def chat_model_params(self) -> dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "url": URL,
            "apikey": WX_APIKEY,
            "project_id": WX_PROJECT_ID,
        }

    @property
    def standard_chat_model_params(self) -> dict[str, Any]:
        return {
            "temperature": 0,
            "params": {
                "include_reasoning": False,
            },
        }

    @pytest.mark.xfail(reason="Supported for vision model.")
    def test_image_inputs(self, model: BaseChatModel) -> None:
        model.watsonx_model._inference.model_id = MODEL_ID_IMAGE  # type: ignore[attr-defined]
        super().test_image_inputs(model)
        model.watsonx_model._inference.model_id = MODEL_ID  # type: ignore[attr-defined]

    @pytest.mark.xfail(reason="Unsupported for v1.")
    @pytest.mark.parametrize("model", [{}, {"output_version": "v1"}], indirect=True)
    def test_agent_loop(self, model: BaseChatModel) -> None:
        super().test_agent_loop(model)

    @pytest.mark.xfail(reason="VCR not set up.")
    def test_stream_time(
        self, model: BaseChatModel, benchmark: BenchmarkFixture, vcr: Cassette
    ) -> None:
        super().test_stream_time(model, benchmark, vcr)
