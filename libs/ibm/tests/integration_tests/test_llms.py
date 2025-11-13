"""Test WatsonxLLM API wrapper.

You'll need to set WATSONX_APIKEY and WATSONX_PROJECT_ID environment variables.
"""

import os

import pytest
from ibm_watsonx_ai import APIClient, Credentials  # type: ignore[import-untyped]
from ibm_watsonx_ai.foundation_models import (  # type: ignore[import-untyped]
    Model,
    ModelInference,
)
from ibm_watsonx_ai.foundation_models.utils.enums import (  # type: ignore[import-untyped]
    DecodingMethods,
)
from ibm_watsonx_ai.metanames import (  # type: ignore[import-untyped]
    GenTextParamsMetaNames,
)
from langchain_core.outputs import LLMResult

from langchain_ibm import WatsonxLLM

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"

MODEL_ID = "ibm/granite-3-3-8b-instruct"

CREATE_WATSONX_LLM_INIT_PARAMETERS = [
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


@pytest.mark.parametrize("init_data", CREATE_WATSONX_LLM_INIT_PARAMETERS)
def test_watsonxllm_init(init_data: dict) -> None:
    watsonxllm = WatsonxLLM(**init_data)

    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_invoke() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_invoke_with_params() -> None:
    parameters = {
        GenTextParamsMetaNames.TEMPERATURE: 0,
        GenTextParamsMetaNames.STOP_SEQUENCES: ["am"],
    }

    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
        params=parameters,
    )
    response = watsonxllm.invoke("Write: 'I am superhero!'\n")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert response.endswith("am")


def test_watsonxllm_invoke_with_params_2() -> None:
    parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 5,
    }

    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.invoke("What color sunflower is?", params=parameters)
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_invoke_with_params_3() -> None:
    parameters_1 = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
    }
    parameters_2 = {
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 5,
    }

    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
        params=parameters_1,
    )
    response = watsonxllm.invoke("What color sunflower is?", params=parameters_2)
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_invoke_with_params_4() -> None:
    parameters_1 = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
    }
    parameters_2 = {
        "temperature": 0.6,
    }

    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
        params=parameters_1,
    )
    response = watsonxllm.invoke("What color sunflower is?", **parameters_2)  # type: ignore[arg-type]
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_invoke_with_params_5_diff() -> None:
    parameters_1 = {
        GenTextParamsMetaNames.TEMPERATURE: 0,
        GenTextParamsMetaNames.STOP_SEQUENCES: ["am"],
    }
    parameters_2 = {
        GenTextParamsMetaNames.TEMPERATURE: 0,
        GenTextParamsMetaNames.STOP_SEQUENCES: ["random_lorem"],
    }

    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
        params=parameters_1,
    )
    response_1 = watsonxllm.invoke("Write: 'I am superhero!'\n")
    print(f"\nResponse 1: {response_1}")
    assert isinstance(response_1, str)
    assert response_1.endswith("am")
    response_2 = watsonxllm.invoke("Write: 'I am superhero!'\n", params=parameters_2)
    print(f"\nResponse 2: {response_2}")
    assert isinstance(response_2, str)
    assert not response_2.endswith("am")


def test_watsonxllm_generate() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.generate(["What color sunflower is?"])
    print(f"\nResponse: {response}")
    response_text = response.generations[0][0].text
    print(f"Response text: {response_text}")
    assert isinstance(response, LLMResult)
    assert len(response_text) > 0


def test_watsonxllm_generate_with_param() -> None:
    parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 5,
    }
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.generate(["What color sunflower is?"], params=parameters)
    print(f"\nResponse: {response}")
    response_text = response.generations[0][0].text
    print(f"Response text: {response_text}")
    assert isinstance(response, LLMResult)
    assert len(response_text) > 0


def test_watsonxllm_generate_with_multiple_prompts() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.generate(
        ["What color sunflower is?", "What color turtle is?"]
    )
    print(f"\nResponse: {response}")
    response_text = response.generations[0][0].text
    print(f"Response text: {response_text}")
    assert isinstance(response, LLMResult)
    assert len(response_text) > 0


def test_watsonxllm_invoke_with_guardrails() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.invoke("What color sunflower is?", guardrails=True)
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_invoke_with_streaming() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
        streaming=True,
    )
    response = watsonxllm.invoke("What color sunflower is?")
    assert isinstance(response, str)


def test_watsonxllm_generate_stream() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.generate(["What color sunflower is?"], stream=True)
    print(f"\nResponse: {response}")
    response_text = response.generations[0][0].text
    print(f"Response text: {response_text}")
    assert isinstance(response, LLMResult)
    assert len(response_text) > 0


def test_watsonxllm_stream() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    stream_response = watsonxllm.stream("What color sunflower is?")

    linked_text_stream = ""
    for chunk in stream_response:
        assert isinstance(chunk, str), (
            f"chunk expect type '{str}', actual '{type(chunk)}'"
        )
        linked_text_stream += chunk
    print(f"Linked text stream: {linked_text_stream}")
    assert linked_text_stream


async def test_watsonxllm_astream() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )

    stream_response = watsonxllm.astream("What color sunflower is?")

    linked_text_stream = ""
    async for chunk in stream_response:
        assert isinstance(chunk, str), (
            f"chunk expect type '{str}', actual '{type(chunk)}'"
        )
        linked_text_stream += chunk

    assert len(linked_text_stream) > 0


def test_watsonxllm_stream_with_kwargs() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    stream_response = watsonxllm.stream("What color sunflower is?", raw_response=True)

    for chunk in stream_response:
        assert isinstance(chunk, str), (
            f"chunk expect type '{str}', actual '{type(chunk)}'"
        )


def test_watsonxllm_stream_with_params() -> None:
    parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "greedy",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 5,
    }
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
        params=parameters,
    )
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")

    stream_response = watsonxllm.stream("What color sunflower is?")

    linked_text_stream = ""
    for chunk in stream_response:
        assert isinstance(chunk, str), (
            f"chunk expect type '{str}', actual '{type(chunk)}'"
        )
        linked_text_stream += chunk
    print(f"Linked text stream: {linked_text_stream}")
    assert response == linked_text_stream, (
        "Linked text stream are not the same as generated text"
    )


def test_watsonxllm_stream_with_params_2() -> None:
    parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "greedy",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 5,
    }
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    stream_response = watsonxllm.stream("What color sunflower is?", params=parameters)

    for chunk in stream_response:
        assert isinstance(chunk, str), (
            f"chunk expect type '{str}', actual '{type(chunk)}'"
        )
        print(chunk)


def test_watsonxllm_stream_with_params_3() -> None:
    parameters_1 = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
    }
    parameters_2 = {
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 5,
    }
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
        params=parameters_1,
    )
    stream_response = watsonxllm.stream("What color sunflower is?", params=parameters_2)

    for chunk in stream_response:
        assert isinstance(chunk, str), (
            f"chunk expect type '{str}', actual '{type(chunk)}'"
        )
        print(chunk)


def test_watsonxllm_stream_with_params_4() -> None:
    parameters_1 = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
    }
    parameters_2 = {
        "temperature": 0.6,
    }
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
        params=parameters_1,
    )
    stream_response = watsonxllm.stream("What color sunflower is?", **parameters_2)  # type: ignore[arg-type]

    for chunk in stream_response:
        assert isinstance(chunk, str), (
            f"chunk expect type '{str}', actual '{type(chunk)}'"
        )
        print(chunk)


def test_watsonxllm_invoke_from_wx_model() -> None:
    model = Model(
        model_id=MODEL_ID,
        credentials={
            "apikey": WX_APIKEY,
            "url": "https://us-south.ml.cloud.ibm.com",
        },
        project_id=WX_PROJECT_ID,
    )
    watsonxllm = WatsonxLLM(watsonx_model=model)
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_invoke_from_wx_model_inference() -> None:
    credentials = Credentials(
        api_key=WX_APIKEY, url="https://us-south.ml.cloud.ibm.com"
    )
    model = ModelInference(
        model_id=MODEL_ID,
        credentials=credentials,
        project_id=WX_PROJECT_ID,
    )
    watsonxllm = WatsonxLLM(watsonx_model=model)
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_invoke_from_wx_model_inference_with_params() -> None:
    parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 10,
        GenTextParamsMetaNames.TEMPERATURE: 0.5,
        GenTextParamsMetaNames.TOP_K: 50,
        GenTextParamsMetaNames.TOP_P: 1,
    }
    model = ModelInference(
        model_id=MODEL_ID,
        credentials={
            "apikey": WX_APIKEY,
            "url": "https://us-south.ml.cloud.ibm.com",
        },
        project_id=WX_PROJECT_ID,
        params=parameters,
    )
    watsonxllm = WatsonxLLM(watsonx_model=model)
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_invoke_from_wx_model_inference_with_params_as_enum() -> None:
    parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: DecodingMethods.GREEDY,
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 10,
        GenTextParamsMetaNames.TEMPERATURE: 0.5,
        GenTextParamsMetaNames.TOP_K: 50,
        GenTextParamsMetaNames.TOP_P: 1,
    }
    model = ModelInference(
        model_id=MODEL_ID,
        credentials={
            "apikey": WX_APIKEY,
            "url": "https://us-south.ml.cloud.ibm.com",
        },
        project_id=WX_PROJECT_ID,
        params=parameters,
    )
    watsonxllm = WatsonxLLM(watsonx_model=model)
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


async def test_watsonx_ainvoke() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = await watsonxllm.ainvoke("What color sunflower is?")
    assert isinstance(response, str)


async def test_watsonx_acall() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = await watsonxllm._acall("what is the color of the grass?")
    assert "green" in response.lower()


async def test_watsonx_agenerate() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = await watsonxllm.agenerate(
        ["What color sunflower is?", "What color turtle is?"]
    )
    assert len(response.generations) > 0
    assert response.llm_output["token_usage"]["completion_tokens"] != 0  # type: ignore[index]


async def test_watsonx_agenerate_with_stream() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = await watsonxllm.agenerate(["What color sunflower is?"], stream=True)
    assert "yellow" in response.generations[0][0].text.lower()


def test_get_num_tokens() -> None:
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    num_tokens = watsonxllm.get_num_tokens("What color sunflower is?")
    assert num_tokens > 0


def test_init_with_client() -> None:
    watsonx_client = APIClient(
        credentials={
            "url": "https://us-south.ml.cloud.ibm.com",
            "apikey": WX_APIKEY,
        }
    )
    watsonxllm = WatsonxLLM(
        model_id=MODEL_ID,
        watsonx_client=watsonx_client,
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_moderations_generate() -> None:
    guardrails_hap_params = {"input": False, "output": True}
    guardrails_pii_params = {"input": False, "output": True}

    watsonxllm = WatsonxLLM(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.generate(
        [
            "Please repeat the words in [], do not trim space.\n"
            "[ I hate this damn world. ]"
        ],
        guardrails=True,
        guardrails_pii_params=guardrails_pii_params,
        guardrails_hap_params=guardrails_hap_params,
    )

    assert response
    assert response.generations[0][0].generation_info["moderations"]  # type: ignore[index]
