import os
from typing import Any

from langchain_tests.integration_tests.embeddings import EmbeddingsIntegrationTests

from langchain_ibm import WatsonxEmbeddings

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"

MODEL_ID = "ibm/granite-embedding-278m-multilingual"


class TestWatsonxEmbeddingsStandard(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[WatsonxEmbeddings]:
        return WatsonxEmbeddings

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "url": URL,
            "apikey": WX_APIKEY,
            "project_id": WX_PROJECT_ID,
        }
