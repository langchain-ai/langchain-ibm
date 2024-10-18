import os
from typing import Type

from langchain_standard_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_ibm import WatsonxEmbeddings

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"

MODEL_ID = "ibm/slate-125m-english-rtrvr"


class TestWatsonxEmbeddingsStandard(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[WatsonxEmbeddings]:
        return WatsonxEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {
            "model_id": MODEL_ID,
            "url": URL,
            "apikey": WX_APIKEY,
            "project_id": WX_PROJECT_ID,
        }
