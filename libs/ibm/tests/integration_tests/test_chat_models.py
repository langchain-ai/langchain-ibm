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
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from langchain_ibm import ChatWatsonx

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://yp-qa.ml.cloud.ibm.com"
MODEL_ID = "ibm/granite-34b-code-instruct"
MODEL_ID_TOOL = "mistralai/mistral-large"


def test_01_generate_chat() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)  # type: ignore[arg-type]
    messages = [
        ("user", "You are a helpful assistant that translates English to French."),
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
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)  # type: ignore[arg-type]
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
        model_id=MODEL_ID,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=parameters_1,
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
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)  # type: ignore[arg-type]
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert response
    for generation in response.generations:
        assert generation[0].text


def test_03_generate_chat_with_few_various_inputs() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)  # type: ignore[arg-type]
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert response
    assert isinstance(response, BaseMessage)
    assert response.content
    assert isinstance(response.content, str)


def test_05_generate_chat_with_stream() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)  # type: ignore[arg-type]
    response = chat.stream("What's the weather in san francisco")
    for chunk in response:
        assert isinstance(chunk.content, str)


def test_05a_invoke_chat_with_streaming() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        streaming=True,
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
        model_id=MODEL_ID,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,
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
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)  # type: ignore[arg-type]
    response = chat.stream("What's the weather in san francisco", params=params)
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
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat
    response = chain.invoke({"text": "Explain the importance of low latency for LLMs."})

    assert response
    assert response.content


def test_10_chaining() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)  # type: ignore[arg-type]
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
        model_id=MODEL_ID,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=parameters,
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


def test_20_bind_tools() -> None:
    params = {"max_tokens": 200}
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather report for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        },
    ]

    llm_with_tools = chat.bind_tools(tools=tools)

    response = llm_with_tools.invoke("what's the weather in san francisco, ca")
    assert isinstance(response, AIMessage)
    assert not response.content
    assert isinstance(response.tool_calls, list)
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]


def test_21a_bind_tools_tool_choice_auto() -> None:
    params = {"max_tokens": 200}
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather report for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        },
    ]

    llm_with_tools = chat.bind_tools(tools=tools, tool_choice="auto")

    response = llm_with_tools.invoke("what's the weather in san francisco, ca")
    assert isinstance(response, AIMessage)
    assert not response.content
    assert isinstance(response.tool_calls, list)
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]


@pytest.mark.skip(reason="Not supported yet")
def test_21b_bind_tools_tool_choice_none() -> None:
    params = {"max_tokens": 200}
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather report for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        },
    ]

    llm_with_tools = chat.bind_tools(tools=tools, tool_choice="none")

    response = llm_with_tools.invoke("what's the weather in san francisco, ca")
    assert isinstance(response, AIMessage)
    assert not response.content
    assert isinstance(response.tool_calls, list)
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]


@pytest.mark.skip(reason="Not supported yet")
def test_21c_bind_tools_tool_choice_required() -> None:
    params = {"max_tokens": 200}
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather report for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        },
    ]

    llm_with_tools = chat.bind_tools(tools=tools, tool_choice="required")

    response = llm_with_tools.invoke("what's the weather in san francisco, ca")
    assert isinstance(response, AIMessage)
    assert not response.content
    assert isinstance(response.tool_calls, list)
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]


def test_22a_bind_tools_tool_choice_as_class() -> None:
    """Test that tool choice is respected."""
    params = {"max_tokens": 200}
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,
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


def test_22b_bind_tools_tool_choice_as_dict() -> None:
    """Test that tool choice is respected just passing in True."""
    params = {"max_tokens": 200}
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,
    )

    class Person(BaseModel):
        name: str
        age: int

    tool_choice = {"type": "function", "function": {"name": "Person"}}

    with_tool = chat.bind_tools([Person], tool_choice=tool_choice)

    result = with_tool.invoke("Erick, 27 years old")
    assert isinstance(result, AIMessage)
    assert result.content == ""  # should just be tool call
    tool_call = result.tool_calls[0]
    assert tool_call["name"] == "Person"
    assert tool_call["args"] == {
        "age": 27,
        "name": "Erick",
    }


def test_23a_bind_tools_list_tool_choice_dict() -> None:
    """Test that tool choice is respected just passing in True."""
    params = {"max_tokens": 200}
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,
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

    tool_choice = {
        "type": "function",
        "function": {
            "name": "add",
        },
    }

    chat_with_tools = chat.bind_tools(tools, tool_choice=tool_choice)

    query = "What is 3 + 12? "
    resp = chat_with_tools.invoke(query)

    assert resp.content == ""


def test_23_bind_tools_list_tool_choice_auto() -> None:
    """Test that tool choice is respected just passing in True."""
    params = {"max_tokens": 200}
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,
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
    chat_with_tools = chat.bind_tools(tools, tool_choice="auto")

    query = "What is 3 + 12? "
    resp = chat_with_tools.invoke(query)
    assert resp.content == ""
    assert len(resp.tool_calls) == 1  # type: ignore
    tool_call = resp.tool_calls[0]  # type: ignore
    assert tool_call["name"] == "add"

    query = "What is 3 * 12? "
    resp = chat_with_tools.invoke(query)
    assert resp.content == ""
    assert len(resp.tool_calls) == 1  # type: ignore
    tool_call = resp.tool_calls[0]  # type: ignore
    assert tool_call["name"] == "multiply"

    query = "Who was the famous painter from Italy?"
    resp = chat_with_tools.invoke(query)
    assert resp.content
    assert len(resp.tool_calls) == 0  # type: ignore


@pytest.mark.skip(reason="Not implemented")
def test_streaming_tool_call() -> None:
    params = {"max_tokens": 200}
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=params,
    )

    class Person(BaseModel):
        name: str
        age: int

    tool_choice = {"type": "function", "function": {"name": "Person"}}

    tool_llm = chat.bind_tools([Person], tool_choice=tool_choice)

    stream_response = tool_llm.stream("Erick, 27 years old")

    additional_kwargs = None
    for chunk in stream_response:
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


@pytest.mark.skip(reason="Not implemented")
def test_streaming_structured_output() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
    )

    class Person(BaseModel):
        name: str
        age: int

    structured_llm = chat.with_structured_output(Person)

    strm_response = structured_llm.stream("Erick, 27 years old")
    chunk_num = 0
    for chunk in strm_response:
        assert chunk_num == 0, "should only have one chunk with model"
        assert isinstance(chunk, Person)
        assert chunk.name == "Erick"
        assert chunk.age == 27
        chunk_num += 1
