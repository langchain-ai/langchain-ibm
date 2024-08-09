import json
import os
from typing import Any

import pytest
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames  # type: ignore
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, SecretStr

from langchain_ibm import ChatWatsonx

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = os.environ.get("WATSONX_URL", "")

MODEL_ID = "mistralai/mixtral-8x7b-instruct-v01"
MISTRAL_LARGE_ID = "mistralai/mistral-large"
LLAMA31_405B_ID = "meta-llama/llama-3-405b-instruct"


def test_01_generate_chat() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    messages = [
        ("system", "You are a helpful assistant that translates English to French."),
        (
            "human",
            "Translate this sentence from English to French. I love programming.",
        ),
    ]
    response = chat.invoke(messages)
    assert response
    assert response.content


def test_01a_generate_chat_with_invoke_params() -> None:
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    params = {
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
    }
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    messages = [
        ("system", "You are a helpful assistant that translates English to French."),
        (
            "human",
            "Translate this sentence from English to French. I love programming.",
        ),
    ]
    response = chat.invoke(messages, params=params)
    assert response
    assert response.content


def test_01b_generate_chat_with_invoke_params() -> None:
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    parameters_1 = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
    }
    parameters_2 = {
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 5,
    }
    chat = ChatWatsonx(
        model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID, params=parameters_1
    )
    messages = [
        ("system", "You are a helpful assistant that translates English to French."),
        (
            "human",
            "Translate this sentence from English to French. I love programming.",
        ),
    ]
    response = chat.invoke(messages, params=parameters_2)
    assert response
    assert response.content


def test_02_generate_chat_with_few_inputs() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert response
    for generation in response.generations:
        assert generation[0].text


def test_03_generate_chat_with_few_various_inputs() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert response
    assert isinstance(response, BaseMessage)
    assert response.content
    assert isinstance(response.content, str)


def test_05_generate_chat_with_stream() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    response = chat.stream("What's the weather in san francisco")
    for chunk in response:
        assert isinstance(chunk.content, str)


def test_05a_invoke_chat_with_streaming() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID, streaming=True
    )
    response = chat.invoke("What's the weather in san francisco")
    assert isinstance(response.content, str)


def test_05_generate_chat_with_stream_with_param() -> None:
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    params = {
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
    }
    chat = ChatWatsonx(
        model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID, params=params
    )
    response = chat.stream("What's the weather in san francisco")
    for chunk in response:
        assert isinstance(chunk.content, str)


def test_05_generate_chat_with_stream_with_param_v2() -> None:
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    params = {
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
    }
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    response = chat.stream(
        "What's the weather in san francisco", params=params)
    for chunk in response:
        assert isinstance(chunk.content, str)


def test_06_chain_invoke() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
    )

    system = "You are a helpful assistant."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", human)])

    chain = prompt | chat
    response = chain.invoke(
        {"text": "Explain the importance of low latency for LLMs."})

    assert response
    assert response.content


def test_10_chaining() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that "
                "translates {input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )
    chain = prompt | chat

    response = chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    assert response
    assert response.content


def test_11_chaining_with_params() -> None:
    parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 5,
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
    }
    chat = ChatWatsonx(
        model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID, params=parameters
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates "
                "{input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )
    chain = prompt | chat

    response = chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    assert response
    assert response.content


def test_20_mistal_large_tool_choice() -> None:
    """Test that tool choice is respected."""
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    params = {GenTextParamsMetaNames.MAX_NEW_TOKENS: 500}
    chat = ChatWatsonx(
        # type: ignore[arg-type]
        model_id=MISTRAL_LARGE_ID, url=URL, project_id=WX_PROJECT_ID, params=params
    )

    class Person(BaseModel):
        name: str
        age: int

    with_tool = chat.bind_tools([Person])

    result = with_tool.invoke("Erick, 27 years old")
    assert isinstance(result, AIMessage)
    assert result.content == ""  # should just be tool call
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call["name"] == "Person"
    assert tool_call["args"] == {
        "age": 27,
        "name": "Erick",
    }


def test_21_tool_mistral_large_choice_bool() -> None:
    """Test that tool choice is respected just passing in True."""
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    params = {GenTextParamsMetaNames.MAX_NEW_TOKENS: 500}
    chat = ChatWatsonx(
        # type: ignore[arg-type]
        model_id=MISTRAL_LARGE_ID, url=URL, project_id=WX_PROJECT_ID, params=params
    )

    class Person(BaseModel):
        name: str
        age: int

    with_tool = chat.bind_tools([Person], tool_choice=True)

    result = with_tool.invoke("Erick, 27 years old")
    assert isinstance(result, AIMessage)
    assert result.content == ""  # should just be tool call
    tool_call = result.tool_calls[0]
    assert tool_call["name"] == "Person"
    assert tool_call["args"] == {
        "age": 27,
        "name": "Erick",
    }


def test_22_mistral_large_tool_invoke() -> None:
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    params = {GenTextParamsMetaNames.MAX_NEW_TOKENS: 500}
    chat = ChatWatsonx(
        model_id=MISTRAL_LARGE_ID,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,  # type: ignore[arg-type]
    )
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Adds a and b."""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiplies a and b."""
        return a * b

    @tool
    def get_word_length(word: str) -> int:
        """Get word length."""
        return len(word)

    tools = [add, multiply, get_word_length]

    chat_with_tools = chat.bind_tools(tools)

    query = "What is 3 + 12? What is 3 + 10?"
    resp = chat_with_tools.invoke(query)

    assert isinstance(resp, AIMessage)
    assert resp.content == ""
    assert len(resp.tool_calls) == 2
    assert resp.tool_calls[0]["name"] == "add"
    assert resp.tool_calls[0]["name"] == "add"

    query = "Who was the famous painter from Italy?"
    resp = chat_with_tools.invoke(query)

    assert isinstance(resp, AIMessage)
    assert resp.content
    assert len(resp.tool_calls) == 0


def test_23_mistral_large_tool_response():
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    messages = [
        ("system", "You are a helpful assistant"),
        ("user", "What is 42 * 2? Respond only with the result.")
    ]

    params = {GenTextParamsMetaNames.MAX_NEW_TOKENS: 500}
    chat = ChatWatsonx(
        model_id=MISTRAL_LARGE_ID,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,  # type: ignore[arg-type]
    )
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Adds a and b."""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiplies a and b."""
        return a * b

    @tool
    def get_word_length(word: str) -> int:
        """Get word length."""
        return len(word)

    tools = [add, multiply, get_word_length]

    chat_with_tools = chat.bind_tools(tools)

    resp = chat_with_tools.invoke(
        ChatPromptTemplate.from_messages(messages).format_messages())

    assert isinstance(resp, AIMessage)
    assert resp.content == ""
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["name"] == "multiply"

    tool_call = resp.tool_calls[0]
    result = int(tool_call["args"]["a"]) * int(tool_call["args"]["b"])

    resp = chat_with_tools.invoke(ChatPromptTemplate.from_messages(messages + [
        resp,
        ToolMessage(content=f"{result}", tool_call_id=tool_call["id"])
    ]).format_messages())

    assert isinstance(resp, AIMessage)
    assert len(resp.tool_calls) == 0
    assert isinstance(resp.content, str)
    assert resp.content.strip() == f"{result}"


def test_2X_llama31_tool_response():
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    messages = [
        ("system", "You are a helpful assistant"),
        ("user", "What is 42 * 2? Respond only with the result.")
    ]

    params = {GenTextParamsMetaNames.MAX_NEW_TOKENS: 500}
    chat = ChatWatsonx(
        model_id=LLAMA31_405B_ID,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,  # type: ignore[arg-type]
    )
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Adds a and b."""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiplies a and b."""
        return a * b

    @tool
    def get_word_length(word: str) -> int:
        """Get word length."""
        return len(word)

    tools = [add, multiply, get_word_length]

    chat_with_tools = chat.bind_tools(tools)

    resp = chat_with_tools.invoke(
        ChatPromptTemplate.from_messages(messages).format_messages())

    assert isinstance(resp, AIMessage)
    assert resp.content == ""
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["name"] == "multiply"

    tool_call = resp.tool_calls[0]
    result = int(tool_call["args"]["a"]) * int(tool_call["args"]["b"])

    resp = chat_with_tools.invoke(ChatPromptTemplate.from_messages(messages + [
        resp,
        ToolMessage(content=f"{result}", tool_call_id=tool_call["id"])
    ]).format_messages())

    assert isinstance(resp, AIMessage)
    assert len(resp.tool_calls) == 0
    assert isinstance(resp.content, str)
    assert resp.content.strip() == f"{result}"


@pytest.mark.skip(reason="Not implemented")
def test_streaming_tool_call() -> None:
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    params = {GenTextParamsMetaNames.MAX_NEW_TOKENS: 500}
    chat = ChatWatsonx(
        model_id=MISTRAL_LARGE_ID,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,  # type: ignore[arg-type]
    )

    class Person(BaseModel):
        name: str
        age: int

    tool_llm = chat.bind_tools([Person])

    # where it calls the tool
    strm = tool_llm.stream("Erick, 27 years old")

    additional_kwargs = None
    for chunk in strm:
        assert isinstance(chunk, AIMessageChunk)
        assert chunk.content == ""
        additional_kwargs = chunk.additional_kwargs

    assert additional_kwargs is not None
    assert "tool_calls" in additional_kwargs
    assert len(additional_kwargs["tool_calls"]) == 1
    assert additional_kwargs["tool_calls"][0]["function"]["name"] == "Person"
    assert json.loads(additional_kwargs["tool_calls"][0]["function"]["arguments"]) == {
        "name": "Erick",
        "age": 27,
    }

    assert isinstance(chunk, AIMessageChunk)
    assert len(chunk.tool_call_chunks) == 1
    tool_call_chunk = chunk.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "Person"
    assert tool_call_chunk["args"] == '{"name": "Erick", "age": 27}'

    # where it doesn't call the tool
    strm = tool_llm.stream("What is 2+2?")
    acc: Any = None
    for chunk in strm:
        assert isinstance(chunk, AIMessageChunk)
        acc = chunk if acc is None else acc + chunk
    assert acc.content != ""
    assert "tool_calls" not in acc.additional_kwargs
