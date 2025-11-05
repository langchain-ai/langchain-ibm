import os

import pytest
from ibm_watsonx_ai import APIClient, Credentials  # type: ignore

from langchain_ibm import WatsonxEmbeddings

URL = "https://us-south.ml.cloud.ibm.com"
WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

MODEL = "ibm/granite-embedding-107m-multilingual"


@pytest.mark.skip("Until not released on us-south, should be skipped.")
class TestEmbeddingsGateway:
    documents = ("What is a generative ai?", "What is a loan and how does it works?")

    def test_embedding_model_gateway_init_credentials(self) -> None:
        watsonx_embedding = WatsonxEmbeddings(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        assert isinstance(watsonx_embedding, WatsonxEmbeddings)

        response = watsonx_embedding.embed_query(text=self.documents[0])
        assert response

    def test_embedding_model_gateway_init_client(self) -> None:
        credentials = Credentials(url=URL, api_key=WX_APIKEY)
        api_client = APIClient(credentials=credentials, project_id=WX_PROJECT_ID)

        watsonx_embedding = WatsonxEmbeddings(model=MODEL, watsonx_client=api_client)
        assert isinstance(watsonx_embedding, WatsonxEmbeddings)

        response = watsonx_embedding.embed_query(text=self.documents[0])
        assert response

    def test_embedding_model_gateway_embed_query(self) -> None:
        watsonx_embedding = WatsonxEmbeddings(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        generate_embedding = watsonx_embedding.embed_query(text=self.documents[0])
        assert isinstance(generate_embedding, list)
        assert isinstance(generate_embedding[0], float)

    async def test_embedding_model_gateway_aembed_query(self) -> None:
        watsonx_embedding = WatsonxEmbeddings(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        generate_embedding = await watsonx_embedding.aembed_query(
            text=self.documents[0]
        )
        assert isinstance(generate_embedding, list)
        assert isinstance(generate_embedding[0], float)

    def test_embedding_model_gateway_embed_documents(self) -> None:
        watsonx_embedding = WatsonxEmbeddings(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        generate_embedding = watsonx_embedding.embed_documents(
            texts=list(self.documents)
        )
        assert generate_embedding
        assert len(generate_embedding) == len(self.documents)
        assert all(isinstance(embedding, list) for embedding in generate_embedding)
        assert all(isinstance(embedding[0], float) for embedding in generate_embedding)
        assert len(generate_embedding[0]) > 0
        assert all(
            len(embedding) == len(generate_embedding[0])
            for embedding in generate_embedding
        )

    async def test_embedding_model_gateway_aembed_documents(self) -> None:
        watsonx_embedding = WatsonxEmbeddings(
            model=MODEL,
            url=URL,
            apikey=WX_APIKEY,
            project_id=WX_PROJECT_ID,
        )
        generate_embedding = await watsonx_embedding.aembed_documents(
            texts=list(self.documents)
        )
        assert generate_embedding
        assert len(generate_embedding) == len(self.documents)
        assert all(isinstance(embedding, list) for embedding in generate_embedding)
        assert all(isinstance(embedding[0], float) for embedding in generate_embedding)
        assert len(generate_embedding[0]) > 0
        assert all(
            len(embedding) == len(generate_embedding[0])
            for embedding in generate_embedding
        )
