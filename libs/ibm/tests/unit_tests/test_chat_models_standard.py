import pytest
from ibm_watsonx_ai import APIClient, Credentials
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests.chat_models import ChatModelUnitTests
from pytest_mock import MockerFixture

from langchain_ibm import ChatWatsonx


class TestWatsonxStandard(ChatModelUnitTests):
    @pytest.fixture(autouse=True)
    def setup_client(self, mocker: MockerFixture) -> None:
        self.client = mocker.Mock(APIClient)
        self.client.credentials = Credentials(api_key="api_key")

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatWatsonx

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_id": "ibm/granite-13b-instruct-v2",
            "validate_model": False,
            "watsonx_client": self.client,
        }
