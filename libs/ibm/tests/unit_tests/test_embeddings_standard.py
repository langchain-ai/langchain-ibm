from typing import Any

import pytest
from ibm_watsonx_ai import APIClient, Credentials
from langchain_tests.unit_tests.embeddings import EmbeddingsUnitTests
from pytest_mock import MockerFixture

from langchain_ibm import WatsonxEmbeddings


class TestWatsonxEmbeddingsStandard(EmbeddingsUnitTests):
    @pytest.fixture(autouse=True)
    def setup_client(self, mocker: MockerFixture) -> None:
        self.client = mocker.Mock(APIClient)
        self.client.credentials = Credentials(url="fake_url", api_key="fake_api_key")

    @property
    def embeddings_class(self) -> type[WatsonxEmbeddings]:
        return WatsonxEmbeddings

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return {
            "model_id": "ibm/granite-13b-instruct-v2",
            "watsonx_client": self.client,
        }
