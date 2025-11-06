import os

import pytest
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models.schema import (
    RerankParameters,
    RerankReturnOptions,
)
from langchain_core.documents import Document

from langchain_ibm import WatsonxRerank

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"

MODEL_ID = "ibm/slate-125m-english-rtrvr"

CREATE_WATSONX_RERANK_INIT_PARAMETERS = [
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


@pytest.mark.parametrize("init_data", CREATE_WATSONX_RERANK_INIT_PARAMETERS)
def test_01_watsonx_rerank_init(init_data: dict) -> None:
    wx_rerank = WatsonxRerank(**init_data)

    assert isinstance(wx_rerank, WatsonxRerank)


def test_01_rerank_init_with_client() -> None:
    wx_client = APIClient(
        credentials={
            "url": URL,
            "apikey": WX_APIKEY,
        }
    )
    wx_rerank = WatsonxRerank(
        model_id=MODEL_ID, project_id=WX_PROJECT_ID, watsonx_client=wx_client
    )
    assert isinstance(wx_rerank, WatsonxRerank)


def test_02_rerank_documents() -> None:
    wx_rerank = WatsonxRerank(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    test_documents = [
        Document(page_content="This is a test document."),
        Document(page_content="Another test document."),
    ]
    test_query = "Test query"
    results = wx_rerank.rerank(test_documents, test_query)
    assert len(results) == 2


def test_02_rerank_documents_with_params() -> None:
    params = RerankParameters(truncate_input_tokens=1)
    test_documents = [
        Document(page_content="This is a test document."),
    ]
    test_query = "Test query"

    wx_rerank = WatsonxRerank(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
    )

    results_1 = wx_rerank.rerank(test_documents, test_query)
    assert len(results_1) == 1

    results_2 = wx_rerank.rerank(test_documents, test_query, params=params)
    assert len(results_2) == 1

    assert results_1[0]["relevance_score"] != results_2[0]["relevance_score"]


def test_03_compress_documents() -> None:
    wx_rerank = WatsonxRerank(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    doc_list = [
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables"
        ", believed to reduce chronic diseases.",
        "Photosynthesis in plants converts light energy into glucose and "
        "produces essential oxygen.",
        "20th-century innovations, from radios to smartphones, centered "
        "on electronic advancements.",
        "Rivers provide water, irrigation, and habitat for aquatic species, "
        "vital for ecosystems.",
        "Apple's conference call to discuss fourth fiscal quarter results and "
        "business updates is scheduled for Thursday, November 2, 2023 at 2:00 "
        "p.m. PT / 5:00 p.m. ET.",
        "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' "
        "endure in literature.",
    ]
    documents = [Document(page_content=x) for x in doc_list]

    result = wx_rerank.compress_documents(
        query="When is the Apple's conference call scheduled?", documents=documents
    )
    assert len(doc_list) == len(result)


def test_04_compress_documents_with_param() -> None:
    params_1 = RerankParameters(
        truncate_input_tokens=1, return_options=RerankReturnOptions(inputs=True)
    )
    params_2 = RerankParameters(truncate_input_tokens=2)
    wx_rerank = WatsonxRerank(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=params_1,
    )
    doc_list = [
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables"
        ", believed to reduce chronic diseases.",
        "Photosynthesis in plants converts light energy into glucose and "
        "produces essential oxygen.",
        "20th-century innovations, from radios to smartphones, centered "
        "on electronic advancements.",
        "Rivers provide water, irrigation, and habitat for aquatic species, "
        "vital for ecosystems.",
        "Apple's conference call to discuss fourth fiscal quarter results and "
        "business updates is scheduled for Thursday, November 2, 2023 at 2:00 "
        "p.m. PT / 5:00 p.m. ET.",
        "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' "
        "endure in literature.",
    ]
    documents = [Document(page_content=x) for x in doc_list]

    result_1 = wx_rerank.compress_documents(
        query="When is the Apple's conference call scheduled?", documents=documents
    )
    assert len(doc_list) == len(result_1)
    result_2 = wx_rerank.compress_documents(
        query="When is the Apple's conference call scheduled?",
        documents=documents,
        params=params_2,
    )
    assert len(doc_list) == len(result_2)

    assert next(
        el.metadata["relevance_score"]
        for el in result_1
        if el.page_content.startswith("20th")
    ) != next(
        el.metadata["relevance_score"]
        for el in result_2
        if el.page_content.startswith("20th")
    )
