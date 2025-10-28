"""Test WatsonxEmbeddings.

You'll need to set WATSONX_APIKEY and WATSONX_PROJECT_ID environment variables.
"""

import os

import pytest
from ibm_watsonx_ai import APIClient  # type: ignore
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings  # type: ignore
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames  # type: ignore

from langchain_ibm import WatsonxEmbeddings

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "ibm/granite-embedding-107m-multilingual"

DOCUMENTS = ["What is a generative ai?", "What is a loan and how does it works?"]

CREATE_WATSONX_EMBEDDINGS_INIT_PARAMETERS = [
    pytest.param(
        {
            "model_id": MODEL_ID,
            "url": URL,
            "api_key": WX_APIKEY,
            "project_id": WX_PROJECT_ID,
        },
        id="only api_key",
    ),
    pytest.param(
        {
            "model_id": MODEL_ID,
            "url": URL,
            "apikey": WX_APIKEY,
            "project_id": WX_PROJECT_ID,
        },
        id="only apikey",
    ),
    pytest.param(
        {
            "model_id": MODEL_ID,
            "url": URL,
            "api_key": WX_APIKEY,
            "apikey": WX_APIKEY,
            "project_id": WX_PROJECT_ID,
        },
        id="api_key and apikey",
    ),
]


@pytest.mark.parametrize("init_data", CREATE_WATSONX_EMBEDDINGS_INIT_PARAMETERS)
def test_watsonx_embeddings_init(init_data: dict) -> None:
    watsonx_embedding = WatsonxEmbeddings(**init_data)

    generate_embedding = watsonx_embedding.embed_documents(texts=DOCUMENTS)
    assert len(generate_embedding) == len(DOCUMENTS)
    assert all(isinstance(el, float) for el in generate_embedding[0])


def test_init_with_client() -> None:
    watsonx_client = APIClient(
        credentials={
            "url": "https://us-south.ml.cloud.ibm.com",
            "apikey": WX_APIKEY,
        },
        project_id=WX_PROJECT_ID,
    )
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID, watsonx_client=watsonx_client
    )
    generate_embedding = watsonx_embedding.embed_documents(texts=DOCUMENTS)
    assert len(generate_embedding) == len(DOCUMENTS)
    assert all(isinstance(el, float) for el in generate_embedding[0])


def test_init_with_embeddings() -> None:
    watsonx_client = APIClient(
        credentials={
            "url": "https://us-south.ml.cloud.ibm.com",
            "apikey": WX_APIKEY,
        },
        project_id=WX_PROJECT_ID,
    )
    embedding = Embeddings(api_client=watsonx_client, model_id=MODEL_ID)

    watsonx_embedding = WatsonxEmbeddings(
        watsonx_embed=embedding,
    )
    generate_embedding = watsonx_embedding.embed_documents(texts=DOCUMENTS)
    assert len(generate_embedding) == len(DOCUMENTS)
    assert all(isinstance(el, float) for el in generate_embedding[0])


def test_01_generate_embed_documents() -> None:
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    generate_embedding = watsonx_embedding.embed_documents(texts=DOCUMENTS)
    assert len(generate_embedding) == len(DOCUMENTS)
    assert all(isinstance(el, float) for el in generate_embedding[0])


async def test_01_generate_aembed_documents() -> None:
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    generate_embedding = await watsonx_embedding.aembed_documents(texts=DOCUMENTS)
    assert len(generate_embedding) == len(DOCUMENTS)
    assert all(isinstance(el, float) for el in generate_embedding[0])


def test_02_generate_embed_documents_with_param() -> None:
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=embed_params,
    )
    generate_embedding = watsonx_embedding.embed_documents(texts=DOCUMENTS)
    assert len(generate_embedding) == len(DOCUMENTS)
    assert all(isinstance(el, float) for el in generate_embedding[0])


def test_03_generate_embed_documents_with_param_in_method() -> None:
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    generate_embedding = watsonx_embedding.embed_documents(
        texts=DOCUMENTS, params=embed_params
    )
    assert len(generate_embedding) == len(DOCUMENTS)
    assert all(isinstance(el, float) for el in generate_embedding[0])


def test_04_generate_embed_documents_with_param_and_concurrency_limit() -> None:
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=embed_params,
    )
    generate_embedding = watsonx_embedding.embed_documents(
        texts=DOCUMENTS, concurrency_limit=9
    )
    assert len(generate_embedding) == len(DOCUMENTS)
    assert all(isinstance(el, float) for el in generate_embedding[0])


def test_10_generate_embed_query() -> None:
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    generate_embedding = watsonx_embedding.embed_query(text=DOCUMENTS[0])
    assert isinstance(generate_embedding, list) and isinstance(
        generate_embedding[0], float
    )


async def test_10_generate_aembed_query() -> None:
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    generate_embedding = await watsonx_embedding.aembed_query(text=DOCUMENTS[0])
    assert isinstance(generate_embedding, list) and isinstance(
        generate_embedding[0], float
    )


def test_11_generate_embed_query_with_params() -> None:
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    generate_embedding = watsonx_embedding.embed_query(
        text=DOCUMENTS[0], params=embed_params
    )
    assert isinstance(generate_embedding, list) and isinstance(
        generate_embedding[0], float
    )


def test_12_generate_embed_query_with_params_and_concurrency_limit() -> None:
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=embed_params,
    )
    generate_embedding = watsonx_embedding.embed_query(
        text=DOCUMENTS[0], concurrency_limit=9
    )
    assert isinstance(generate_embedding, list) and isinstance(
        generate_embedding[0], float
    )


def test_20_generate_embed_query_with_client_initialization() -> None:
    watsonx_client = APIClient(
        credentials={
            "url": URL,
            "apikey": WX_APIKEY,
        }
    )
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID, project_id=WX_PROJECT_ID, watsonx_client=watsonx_client
    )
    generate_embedding = watsonx_embedding.embed_query(text=DOCUMENTS[0])
    assert isinstance(generate_embedding, list) and isinstance(
        generate_embedding[0], float
    )
