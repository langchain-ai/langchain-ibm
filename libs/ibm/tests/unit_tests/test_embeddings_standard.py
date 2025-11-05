from ibm_watsonx_ai import APIClient, Credentials  # type: ignore[import-untyped]
from ibm_watsonx_ai.service_instance import (  # type: ignore[import-untyped]
    ServiceInstance,
)
from langchain_tests.unit_tests.embeddings import EmbeddingsUnitTests

from langchain_ibm import WatsonxEmbeddings

client = APIClient.__new__(APIClient)
client.CLOUD_PLATFORM_SPACES = True
client.ICP_PLATFORM_SPACES = True
credentials = Credentials(api_key="api_key")
client.credentials = credentials
client.service_instance = ServiceInstance.__new__(ServiceInstance)
client._httpx_client = None
client._async_httpx_client = None


class TestWatsonxEmbeddingsStandard(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> type[WatsonxEmbeddings]:
        return WatsonxEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {
            "model_id": "ibm/granite-13b-instruct-v2",
            "watsonx_client": client,
        }
