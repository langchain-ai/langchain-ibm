from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
from ibm_watsonx_ai.service_instance import ServiceInstance  # type: ignore
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests.chat_models import ChatModelUnitTests

from langchain_ibm import ChatWatsonx

client = APIClient.__new__(APIClient)
client.CLOUD_PLATFORM_SPACES = True
client.ICP_PLATFORM_SPACES = True
credentials = Credentials(api_key="api_key")
client.credentials = credentials
client.service_instance = ServiceInstance.__new__(ServiceInstance)
client.default_space_id = None
client.default_project_id = None
client._httpx_client = None
client._async_httpx_client = None


class TestWatsonxStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatWatsonx

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_id": "ibm/granite-13b-instruct-v2",
            "validate_model": False,
            "watsonx_client": client,
        }
