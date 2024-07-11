import json
import os
from typing import Any

import pytest
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames  # type: ignore
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel

from langchain_ibm import ChatWatsonx

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "mistralai/mixtral-8x7b-instruct-v01"


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


def test_02_generate_chat_with_few_inputs() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert response


def test_03_generate_chat_with_few_various_inputs() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_05_generate_chat_with_stream() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    response = chat.stream("What's the weather in san francisco")
    for chunk in response:
        assert isinstance(chunk.content, str)


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


def test_20_tool_choice() -> None:
    """Test that tool choice is respected."""
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    params = {GenTextParamsMetaNames.MAX_NEW_TOKENS: 500}
    chat = ChatWatsonx(
        model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID, params=params
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


def test_21_tool_choice_bool() -> None:
    """Test that tool choice is respected just passing in True."""
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    params = {GenTextParamsMetaNames.MAX_NEW_TOKENS: 500}
    chat = ChatWatsonx(
        model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID, params=params
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


def test_22_tool_invoke() -> None:
    """Test that tool choice is respected just passing in True."""
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    params = {GenTextParamsMetaNames.MAX_NEW_TOKENS: 500}
    chat = ChatWatsonx(
        model_id=MODEL_ID,
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
    resp = chat_with_tools.invoke(query, params=params)

    assert resp.content == ""

    query = "Who was the famous painter from Italy?"
    resp = chat_with_tools.invoke(query, params=params)

    assert resp.content


@pytest.mark.skip(reason="Not implemented")
def test_streaming_tool_call() -> None:
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

    params = {GenTextParamsMetaNames.MAX_NEW_TOKENS: 500}
    chat = ChatWatsonx(
        model_id=MODEL_ID,
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
