import json
import os
import re
from typing import Any, Literal, cast

import pytest
from ibm_watsonx_ai.foundation_models.schema import (  # type: ignore[import-untyped]
    TextChatParameters,
    TextChatResponseFormat,
)
from ibm_watsonx_ai.metanames import (  # type: ignore[import-untyped]
    GenTextParamsMetaNames,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import (
    BaseModel,
    SecretStr,
)

from langchain_ibm import ChatWatsonx, WatsonxToolkit

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = SecretStr(secret_value="https://us-south.ml.cloud.ibm.com")

MODEL_ID = "ibm/granite-3-3-8b-instruct"
MODEL_ID_TOOL = "ibm/granite-3-3-8b-instruct"
MODEL_ID_TOOL_2 = "meta-llama/llama-3-3-70b-instruct"
MODEL_ID_REASONING_CONTENT = "openai/gpt-oss-120b"

PARAMS_WITH_MAX_TOKENS = {"max_tokens": 20}


@pytest.mark.token_check
def test_chat_invoke() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
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


def test_chat_invoke_with_reasoning_content() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_REASONING_CONTENT,
        url=URL,
        project_id=WX_PROJECT_ID,
        params={
            "include_reasoning": True,
            "reasoning_effort": "low",
        },
    )
    messages = [("human", "Say hello!")]
    response = chat.invoke(messages)
    assert response
    assert response.content
    assert response.additional_kwargs.get("reasoning_content")

    response_2 = chat.invoke(messages, params={"include_reasoning": False})
    assert response_2
    assert response_2.content
    assert not response_2.additional_kwargs.get("reasoning_content")


def test_chat_invoke_with_params_as_dict_in_invoke() -> None:
    params = {"max_tokens": 10}
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
    print(response.content)


def test_chat_invoke_with_params_as_object_in_invoke() -> None:
    params = TextChatParameters(max_tokens=10)
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
    print(response.content)


def test_chat_invoke_with_params_as_object_in_constructor() -> None:
    params = TextChatParameters(max_tokens=10)
    chat = ChatWatsonx(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=params,
    )
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
    print(response.content)


def test_chat_invoke_with_invoke_params() -> None:
    parameters_1 = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
    }
    parameters_2 = {
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 5,
    }
    chat = ChatWatsonx(
        model_id=MODEL_ID,
        url=URL,
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


def test_chat_generate_with_few_inputs() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=PARAMS_WITH_MAX_TOKENS,
    )
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert response
    for generation in response.generations:
        assert generation[0].text


def test_chat_generate_with_reasoning_content() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_REASONING_CONTENT,
        url=URL,
        project_id=WX_PROJECT_ID,
        params={
            "include_reasoning": True,
            "reasoning_effort": "low",
        },
    )
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert response
    for generation in response.generations:
        assert generation[0].text
        assert generation[0].message  # type: ignore[attr-defined]
        assert "reasoning_content" in generation[0].message.additional_kwargs  # type: ignore[attr-defined]
        assert generation[0].message.additional_kwargs["reasoning_content"]  # type: ignore[attr-defined]

    response_2 = chat.generate(
        [[message], [message]], params={"include_reasoning": False}
    )
    assert response_2
    for generation in response_2.generations:
        assert generation[0].text
        assert generation[0].message  # type: ignore[attr-defined]
        assert "reasoning_content" not in generation[0].message.additional_kwargs  # type: ignore[attr-defined]


async def test_chat_agenerate() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert response
    for generation in response.generations:
        assert generation[0].text


def test_chat_invoke_with_few_various_inputs() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=PARAMS_WITH_MAX_TOKENS,
    )
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert response
    assert isinstance(response, BaseMessage)
    assert response.content
    assert isinstance(response.content, str)


async def test_chat_ainvoke() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    messages = [
        ("user", "You are a helpful assistant that translates English to French."),
        (
            "human",
            "Translate this sentence from English to French. I love programming.",
        ),
    ]
    response = await chat.ainvoke(messages)
    assert response
    assert response.content


def test_chat_stream() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=PARAMS_WITH_MAX_TOKENS,
    )
    response = chat.stream("What's the weather in san francisco")
    for chunk in response:
        assert isinstance(chunk.content, str)


def test_chat_stream_with_reasoning_content() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_REASONING_CONTENT,
        url=URL,
        project_id=WX_PROJECT_ID,
        params={
            "include_reasoning": True,
            "reasoning_effort": "low",
        },
    )
    response = chat.stream("hello")

    reasoning_content = ""

    for chunk in response:
        assert isinstance(chunk.content, str)
        reasoning_content += chunk.additional_kwargs.get("reasoning_content", "")

    assert reasoning_content


async def test_chat_astream() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    messages = [
        ("user", "You are a helpful assistant that translates English to French."),
        (
            "human",
            "Translate this sentence from English to French. I love programming.",
        ),
    ]
    num_tokens = 0

    async for chunk in chat.astream(messages):
        assert chunk is not None
        assert isinstance(chunk, AIMessageChunk)
        num_tokens += len(chunk.content)
    assert num_tokens > 0


def test_chat_invoke_with_streaming() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
        streaming=True,
        params=PARAMS_WITH_MAX_TOKENS,
    )
    response = chat.invoke("What's the weather in san francisco")
    assert isinstance(response.content, str)


def test_chat_stream_with_param_in_constructor() -> None:
    params = TextChatParameters(max_tokens=10)
    chat = ChatWatsonx(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=params,
    )
    response = chat.stream("What's the weather in san francisco")
    for chunk in response:
        assert isinstance(chunk.content, str)


def test_chat_stream_with_param_in_method() -> None:
    params = TextChatParameters(max_tokens=10)
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    response = chat.stream("What's the weather in san francisco", params=params)
    for chunk in response:
        assert isinstance(chunk.content, str)


def test_chain_invoke() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=PARAMS_WITH_MAX_TOKENS,
    )

    system = "You are a helpful assistant."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat
    response = chain.invoke({"text": "Explain the importance of low latency for LLMs."})

    assert response
    assert response.content


def test_chain_invoke_2() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=PARAMS_WITH_MAX_TOKENS,
    )
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


def test_chat_bind_tools() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
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


def test_chat_bind_tools_tool_choice_auto() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
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


@pytest.mark.xfail(reason="Not supported yet")
def test_chat_bind_tools_tool_choice_none() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
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


@pytest.mark.xfail(reason="Not supported yet")
def test_chat_bind_tools_tool_choice_required() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
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


def test_chat_bind_tools_tool_choice_as_class() -> None:
    """Test that tool choice is respected."""
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        params={"temperature": 0},
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


def test_chat_bind_tools_tool_choice_as_dict() -> None:
    """Test that tool choice is respected just passing in True."""
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        params={"temperature": 0},
    )

    class Person(BaseModel):
        name: str
        age: int

    tool_choice = {"type": "function", "function": {"name": "Person"}}

    with_tool = chat.bind_tools([Person], tool_choice=tool_choice)

    result = with_tool.invoke("Erick, 27 years old. Make sure to use correct name")
    assert isinstance(result, AIMessage)
    assert result.content == ""  # should just be tool call
    tool_call = result.tool_calls[0]
    assert tool_call["name"] == "Person"
    assert tool_call["args"] == {
        "age": 27,
        "name": "Erick",
    }


def test_chat_bind_tools_list_tool_choice_dict() -> None:
    """Test that tool choice is respected just passing in True."""
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL_2,
        url=URL,
        project_id=WX_PROJECT_ID,
        params={"temperature": 0},
    )

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


def test_chat_bind_tools_list_tool_choice_auto() -> None:
    """Test that tool choice is respected just passing in True."""
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL_2,
        url=URL,
        project_id=WX_PROJECT_ID,
        params={"temperature": 0},
    )

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


def test_chat_bind_tools_with_watsonx_tools() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    toolkit = WatsonxToolkit(
        url=URL,
    )
    weather_tool = toolkit.get_tool("Weather")

    tools = [weather_tool]

    llm_with_tools = chat.bind_tools(tools=tools)

    response = llm_with_tools.invoke("what's the weather in san francisco, ca")
    assert isinstance(response, AIMessage)
    assert not response.content
    assert isinstance(response.tool_calls, list)
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "Weather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]


def test_chat_bind_tools_with_watsonx_tools_tool_choice_auto() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
    )

    toolkit = WatsonxToolkit(
        url=URL,
    )
    weather_tool = toolkit.get_tool("Weather")

    tools = [weather_tool]

    llm_with_tools = chat.bind_tools(tools=tools, tool_choice="auto")

    response = llm_with_tools.invoke("what's the weather in san francisco, ca")
    assert isinstance(response, AIMessage)
    assert not response.content
    assert isinstance(response.tool_calls, list)
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "Weather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]


def test_chat_bind_tools_with_watsonx_tools_tool_choice_as_dict() -> None:
    """Test that tool choice is respected just passing in True."""
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL_2,
        url=URL,
        project_id=WX_PROJECT_ID,
        params={"temperature": 0},
    )

    toolkit = WatsonxToolkit(
        url=URL,
    )
    weather_tool = toolkit.get_tool("Weather")

    tools = [weather_tool]

    tool_choice = {"type": "function", "function": {"name": "Weather"}}

    with_tool = chat.bind_tools(tools, tool_choice=tool_choice)

    result = with_tool.invoke("What's the weather in Boston?")
    assert isinstance(result, AIMessage)
    assert result.content == ""  # should just be tool call
    tool_call = result.tool_calls[0]
    assert tool_call["name"] == "Weather"
    assert tool_call["args"] == {
        "location": "Boston",
    }


def test_chat_bind_tools_with_watsonx_tools_list_tool_choice_auto() -> None:
    """Test that tool choice is respected just passing in True."""
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        params={"temperature": 0},
    )

    toolkit = WatsonxToolkit(
        url=URL,
    )
    weather_tool = toolkit.get_tool("Weather")
    google_search_tool = toolkit.get_tool("GoogleSearch")

    tools = [weather_tool, google_search_tool]
    chat_with_tools = chat.bind_tools(tools, tool_choice="auto")

    query = "What is the weather in Boston?"
    resp = chat_with_tools.invoke(query)
    assert resp.content == ""
    assert len(resp.tool_calls) == 1  # type: ignore
    tool_call = resp.tool_calls[0]  # type: ignore
    assert tool_call["name"] == "Weather"

    query = "Search for IBM"
    resp = chat_with_tools.invoke(query)
    assert resp.content == ""
    assert len(resp.tool_calls) == 1  # type: ignore
    tool_call = resp.tool_calls[0]  # type: ignore
    assert tool_call["name"] == "GoogleSearch"

    query = "How are you doing?"
    resp = chat_with_tools.invoke(query)
    assert resp.content
    assert len(resp.tool_calls) == 0  # type: ignore


def test_chat_bind_tools_with_watsonx_tools_list_tool_choice_dict() -> None:
    """Test that tool choice is respected just passing in True."""
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        params={"temperature": 0},
    )

    toolkit = WatsonxToolkit(
        url=URL,
    )
    weather_tool = toolkit.get_tool("Weather")
    google_search_tool = toolkit.get_tool("GoogleSearch")

    tools = [weather_tool, google_search_tool]

    tool_choice = {
        "type": "function",
        "function": {
            "name": "GoogleSearch",
        },
    }

    chat_with_tools = chat.bind_tools(tools, tool_choice=tool_choice)

    query = "What is the weather in Boston?"
    resp = chat_with_tools.invoke(query)
    assert resp.content == ""
    assert len(resp.tool_calls) == 1  # type: ignore
    tool_call = resp.tool_calls[0]  # type: ignore
    assert tool_call["name"] == "GoogleSearch"


def test_chat_with_json_mode() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    response = chat.invoke(
        "Return this as json: {'a': 1}",
        params={"response_format": {"type": "json_object"}},
    )
    assert isinstance(response.content, str)
    assert json.loads(response.content) == {"a": 1}

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in chat.stream(
        "Return this as json: {'a': 1}",
        params={"response_format": {"type": "json_object"}},
    ):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, str)
    assert json.loads(full.content) == {"a": 1}


async def test_chat_with_json_mode_async() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    response = await chat.ainvoke(
        "Return this as json: {'a': 1}",
        params={"response_format": {"type": "json_object"}},
    )
    assert isinstance(response.content, str)
    assert json.loads(response.content) == {"a": 1}

    # Test streaming
    full: BaseMessageChunk | None = None
    async for chunk in chat.astream(
        "Return this as json: {'a': 1}",
        params={"response_format": {"type": "json_object"}},
    ):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, str)
    assert json.loads(full.content) == {"a": 1}


@pytest.mark.xfail(reason="Not implemented")
def test_chat_streaming_tool_call() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
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


def test_chat_streaming_multiple_tool_call() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        temperature=0,
    )

    @tool("search")
    def search(query: str) -> list[str]:  # noqa: ARG001
        """Call to search the web for capital of countries"""
        return ["capital of america is washington D.C."]

    @tool("get_weather")
    def get_weather(city: Literal["nyc"]) -> str:
        """Use this to get weather information."""
        if city == "nyc":
            return "It might be cloudy in nyc"
        error_msg = "Unknown city"  # type: ignore[unreachable]
        raise ValueError(error_msg)

    tools = [search, get_weather]
    tools_name = {el.name for el in tools}

    tool_llm = chat.bind_tools(tools)

    stream_response = tool_llm.stream(
        "What is the weather in the NY and what is capital of USA?"
    )

    ai_message = None

    for chunk in stream_response:
        if ai_message is None:
            ai_message = chunk
        else:
            ai_message += chunk  # type: ignore[assignment]
        print(chunk.id, type(chunk.id))
        assert isinstance(chunk, AIMessageChunk)
        assert chunk.content == ""

    ai_message = cast("AIMessageChunk", ai_message)
    assert ai_message.response_metadata.get("finish_reason") == "tool_calls"
    assert ai_message.response_metadata.get("model_name") == MODEL_ID_TOOL
    assert ai_message.id is not None

    # additional_kwargs
    assert ai_message.additional_kwargs is not None
    assert "tool_calls" in ai_message.additional_kwargs
    assert len(ai_message.additional_kwargs["tool_calls"]) == 2
    assert {
        el["function"]["name"] for el in ai_message.additional_kwargs["tool_calls"]
    } == tools_name

    # tool_calls
    assert all(el["id"] is not None for el in ai_message.tool_calls)
    assert all(el["type"] == "tool_call" for el in ai_message.tool_calls)
    assert {el["name"] for el in ai_message.tool_calls} == tools_name

    generated_tools_args = [{"city": "nyc"}, {"query": "capital of USA"}]
    assert {next(iter(el["args"].keys())) for el in ai_message.tool_calls} == {
        next(iter(el.keys())) for el in generated_tools_args
    }

    # tool_call_chunks
    predicted_tool_call_chunks = []
    for i, el in enumerate(ai_message.tool_calls):
        el |= {"type": "tool_call_chunk"}  # type: ignore[typeddict-item]
        el["args"] = json.dumps(el["args"])  # type: ignore[typeddict-item]
        el |= {"index": i}  # type: ignore[misc]
        predicted_tool_call_chunks.append(el)

    assert ai_message.tool_call_chunks == predicted_tool_call_chunks  # type: ignore[comparison-overlap]
    assert (
        json.loads(
            ai_message.additional_kwargs["tool_calls"][0]["function"]["arguments"]
        )
        == generated_tools_args[0]
    )
    assert (
        json.loads(
            ai_message.additional_kwargs["tool_calls"][1]["function"]["arguments"]
        )
        == generated_tools_args[1]
    )

    # TODO: these tests should works when usage field will be fixed
    # assert ai_message.usage_metadata is not None


def test_chat_structured_output_function_calling() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        temperature=0,
    )
    schema = {
        "title": "AnswerWithJustification",
        "description": (
            "An answer to the user question along with justification for the answer."
        ),
        "type": "object",
        "properties": {
            "answer": {"title": "Answer", "type": "string"},
            "justification": {"title": "Justification", "type": "string"},
        },
        "required": ["answer", "justification"],
    }
    structured_llm = chat.with_structured_output(schema, method="function_calling")
    result = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )
    assert isinstance(result, dict)
    assert "answer" in result and "justification" in result


def test_chat_structured_output_json_schema() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        temperature=0,
    )
    schema = {
        "title": "AnswerWithJustification",
        "description": (
            "An answer to the user question along with justification for the answer."
        ),
        "type": "object",
        "properties": {
            "answer": {"title": "Answer", "type": "string"},
            "justification": {"title": "Justification", "type": "string"},
        },
        "required": ["answer", "justification"],
    }
    structured_llm = chat.with_structured_output(schema, method="json_schema")
    result = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )
    assert isinstance(result, dict)
    assert "answer" in result and "justification" in result


def test_chat_streaming_structured_output_function_calling() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL_2,
        url=URL,
        project_id=WX_PROJECT_ID,
    )

    class Person(BaseModel):
        name: str
        age: int

    structured_llm = chat.with_structured_output(Person)
    stream_response = structured_llm.stream("Erick, 123456 years old")

    for chunk in stream_response:
        assert isinstance(chunk, Person)
        assert chunk.name in "Erick"
        assert chunk.age == 123456


###################################
# --------Parameter-Tests-------- #
###################################

prompt_1 = "Say: 'Hello, My name is Erick!'"


@pytest.mark.parametrize(("params_1", "expected"), [(None, {}), ({}, {})])
def test_init_with_params_1(params_1: Any, expected: Any) -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=params_1,
    )
    assert chat.params == expected


@pytest.mark.parametrize(
    ("params_1", "expected"),
    [
        ({"max_tokens": 10}, {"max_tokens": 10}),
        (TextChatParameters(max_tokens=10), {"max_tokens": 10}),
    ],
)
def test_init_with_params_2(params_1: Any, expected: Any) -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=params_1,
    )
    assert chat.params == expected


@pytest.mark.parametrize(
    ("params_1", "expected"), [({"max_tokens": 10}, {"max_tokens": 10})]
)
def test_init_with_params_3(params_1: Any, expected: Any) -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        **params_1,
    )
    assert chat.params == expected


@pytest.mark.parametrize(
    ("params_1", "params_2", "expected"),
    [
        (
            {"max_tokens": 10},
            {"temperature": 0.5},
            {"max_tokens": 10, "temperature": 0.5},
        ),
        (
            TextChatParameters(max_tokens=10),
            {"temperature": 0.5},
            {"max_tokens": 10, "temperature": 0.5},
        ),
    ],
)
def test_init_with_params_4(params_1: Any, params_2: Any, expected: Any) -> None:
    params_1 = {"max_tokens": 10}
    params_2 = {"temperature": 0.5}
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=params_1,
        **params_2,
    )
    assert chat.params == expected


@pytest.mark.parametrize(
    ("params_1", "params_2"),
    [
        (
            {"max_tokens": 10},
            {"max_tokens": 20},
        ),
        (
            TextChatParameters(max_tokens=10),
            {"max_tokens": 20},
        ),
    ],
)
def test_init_with_params_5(params_1: Any, params_2: Any) -> None:
    pattern = re.escape(
        "Duplicate parameters found in params and keyword arguments: ['max_tokens']"
    )

    with pytest.raises(ValueError, match=pattern) as e:
        ChatWatsonx(
            model_id=MODEL_ID_TOOL,
            url=URL,
            project_id=WX_PROJECT_ID,
            params=params_1,
            **params_2,
        )
    assert (
        "Duplicate parameters found in params and keyword arguments: ['max_tokens']"
        in str(e.value)
    )


@pytest.mark.parametrize(("params_1", "expected"), [(None, {}), ({}, {})])
def test_invoke_with_params_1(params_1: Any, expected: Any) -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    resp = chat.invoke(prompt_1, params=params_1)
    completion_tokens = resp.response_metadata.get("token_usage", {}).get(
        "completion_tokens"
    )

    assert chat.params == expected
    assert completion_tokens > 0


@pytest.mark.parametrize(
    ("params_1", "expected"),
    [
        ({"max_tokens": 5}, 5),
        (TextChatParameters(max_tokens=5), 5),
        (TextChatParameters(max_completion_tokens=5), 5),
    ],
)
def test_invoke_with_params_2(params_1: Any, expected: Any) -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    resp = chat.invoke(prompt_1, params=params_1)
    completion_tokens = resp.response_metadata.get("token_usage", {}).get(
        "completion_tokens"
    )

    assert chat.params == {}
    assert completion_tokens == expected


@pytest.mark.parametrize(
    ("params_1", "params_2"),
    [
        (
            {"max_completion_tokens": 10},
            {"temperature": 1},
        ),
        (
            TextChatParameters(max_completion_tokens=10),
            {"temperature": 1},
        ),
    ],
)
def test_invoke_with_params_3(params_1: Any, params_2: Any) -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
    )

    resp = chat.invoke(prompt_1, params=params_1, **params_2)

    assert resp

    completion_tokens = resp.response_metadata.get("token_usage", {}).get(
        "completion_tokens"
    )
    assert completion_tokens > 0


@pytest.mark.parametrize(
    ("params_1", "params_2"),
    [
        ({"max_completion_tokens": 10}, {"type": "text"}),
        (
            TextChatParameters(max_completion_tokens=10),
            TextChatResponseFormat(type="text"),
        ),
    ],
)
def test_invoke_with_params_4(params_1: Any, params_2: Any) -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
    )

    resp = chat.invoke(prompt_1, params=params_1, response_format=params_2)

    assert resp

    completion_tokens = resp.response_metadata.get("token_usage", {}).get(
        "completion_tokens"
    )
    assert completion_tokens > 0


@pytest.mark.parametrize(
    ("params_1_a", "params_1_b", "params_2_a", "params_2_b", "expected_tokens"),
    [
        (
            {"max_tokens": 5},
            {"max_tokens": 10},
            {"logprobs": False},
            {"logprobs": True},
            5,
        )
    ],
)
def test_invoke_with_params_5(
    params_1_a: Any,
    params_1_b: Any,
    params_2_a: Any,
    params_2_b: Any,
    expected_tokens: Any,
) -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    resp_1 = chat.invoke(prompt_1, params=params_1_a, **params_2_a)
    completion_tokens = resp_1.response_metadata.get("token_usage", {}).get(
        "completion_tokens"
    )
    logprobs = resp_1.response_metadata.get("logprobs")

    assert chat.params == {}
    assert completion_tokens == expected_tokens
    assert not logprobs

    resp_2 = chat.invoke(prompt_1, params=params_1_a, **params_2_b)
    completion_tokens = resp_2.response_metadata.get("token_usage", {}).get(
        "completion_tokens"
    )
    logprobs = resp_2.response_metadata.get("logprobs")

    assert chat.params == {}
    assert completion_tokens == expected_tokens
    assert logprobs

    resp_3 = chat.invoke(prompt_1, **params_1_b, **params_2_b)
    completion_tokens = resp_3.response_metadata.get("token_usage", {}).get(
        "completion_tokens"
    )
    logprobs = resp_3.response_metadata.get("logprobs")

    assert chat.params == {}
    assert 7 < completion_tokens < 11
    assert logprobs


@pytest.mark.parametrize(
    ("params_1", "params_2"),
    [
        (
            {"max_tokens": 5},
            {"max_tokens": 20},
        ),
        (
            TextChatParameters(max_tokens=5),
            {"max_tokens": 20},
        ),
    ],
)
def test_invoke_with_params_6(params_1: Any, params_2: Any) -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    with pytest.raises(ValueError) as e:
        chat.invoke(prompt_1, params=params_1, **params_2)

    assert (
        "Duplicate parameters found in params and keyword arguments: ['max_tokens']"
        in str(e.value)
    )


@pytest.mark.parametrize(
    ("params_1", "params_2", "params_3"),
    [
        ({"max_tokens": 5, "logprobs": False}, {"max_tokens": 10}, {"logprobs": True}),
        (
            TextChatParameters(max_tokens=5, logprobs=False),
            {"max_tokens": 10},
            {"logprobs": True},
        ),
    ],
)
def test_invoke_with_params_7(params_1: Any, params_2: Any, params_3: Any) -> None:
    params_1 = {"max_tokens": 5, "logprobs": False}
    params_2 = {"max_tokens": 10}
    params_3 = {"logprobs": True}
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    pattern = (
        r"(?=.*Duplicate parameters found in params and keyword arguments: )"
        r"(?=.*'logprobs')(?=.*'max_tokens')"
    )
    with pytest.raises(ValueError, match=pattern) as e:
        chat.invoke(prompt_1, params=params_1, **params_2, **params_3)

    assert (
        "Duplicate parameters found in params and keyword arguments: " in str(e.value)
        and "'logprobs'" in str(e.value)
        and "'max_tokens'" in str(e.value)
    )


@pytest.mark.parametrize(
    ("params_1", "expected"),
    [
        ({"max_tokens": 11}, {"max_tokens": 11}),
        (TextChatParameters(max_tokens=11), {"max_tokens": 11}),
    ],
)
def test_init_and_invoke_with_params_1(params_1: Any, expected: Any) -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=params_1,
    )
    resp = chat.invoke(prompt_1, params=params_1)
    completion_tokens = resp.response_metadata.get("token_usage", {}).get(
        "completion_tokens"
    )
    assert chat.params == expected
    assert 7 < completion_tokens <= 11


@pytest.mark.parametrize(
    (
        "params_1_a",
        "params_1_b",
        "params_1_c",
        "expected_tokens_1",
        "expected_tokens_2",
        "expected_tokens_3",
    ),
    [
        ({"max_tokens": 4}, {"max_tokens": 5}, {"max_tokens": 6}, 5, 6, 4),
    ],
)
def test_init_and_invoke_with_params_2(
    params_1_a: Any,
    params_1_b: Any,
    params_1_c: Any,
    expected_tokens_1: Any,
    expected_tokens_2: Any,
    expected_tokens_3: Any,
) -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=params_1_a,
    )
    resp_1 = chat.invoke(prompt_1, params=params_1_b)
    completion_tokens = resp_1.response_metadata.get("token_usage", {}).get(
        "completion_tokens"
    )
    assert chat.params == params_1_a
    assert completion_tokens == expected_tokens_1

    resp_2 = chat.invoke(prompt_1, **params_1_c)
    completion_tokens = resp_2.response_metadata.get("token_usage", {}).get(
        "completion_tokens"
    )
    assert chat.params == params_1_a
    assert completion_tokens == expected_tokens_2

    resp_2 = chat.invoke(prompt_1)
    completion_tokens = resp_2.response_metadata.get("token_usage", {}).get(
        "completion_tokens"
    )
    assert chat.params == params_1_a
    assert completion_tokens == expected_tokens_3


@pytest.mark.parametrize(
    (
        "params_1_a",
        "params_1_b",
        "params_2_a",
        "params_2_b",
        "expected_tokens_1",
        "expected_tokens_2",
    ),
    [
        (
            {"max_tokens": 2},
            {"max_tokens": 5},
            {"logprobs": False},
            {"logprobs": True},
            5,
            2,
        )
    ],
)
def test_init_and_invoke_with_params_3(
    params_1_a: Any,
    params_1_b: Any,
    params_2_a: Any,
    params_2_b: Any,
    expected_tokens_1: Any,
    expected_tokens_2: Any,
) -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=params_1_a,
        **params_2_a,
    )
    resp_1 = chat.invoke(prompt_1, params=params_1_b, **params_2_b)
    completion_tokens = resp_1.response_metadata.get("token_usage", {}).get(
        "completion_tokens"
    )
    logprobs = resp_1.response_metadata.get("logprobs")

    assert chat.params == params_1_a | params_2_a
    assert completion_tokens == expected_tokens_1
    assert logprobs

    resp_2 = chat.invoke(prompt_1)
    completion_tokens = resp_2.response_metadata.get("token_usage", {}).get(
        "completion_tokens"
    )
    logprobs = resp_2.response_metadata.get("logprobs")

    assert chat.params == params_1_a | params_2_a
    assert completion_tokens == expected_tokens_2
    assert not logprobs


@pytest.mark.parametrize(
    ("params_1_a", "params_1_b", "params_1_c"),
    [
        (
            {"max_tokens": 4},
            {"max_tokens": 5},
            {"max_tokens": 6},
        ),
    ],
)
def test_init_and_invoke_with_params_4(
    params_1_a: Any, params_1_b: Any, params_1_c: Any
) -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID_TOOL,
        url=URL,
        project_id=WX_PROJECT_ID,
        params=params_1_a,
    )

    pattern = re.escape(
        "Duplicate parameters found in params and keyword arguments: ['max_tokens']"
    )

    with pytest.raises(ValueError, match=pattern) as e:
        chat.invoke(prompt_1, params=params_1_b, **params_1_c)

    assert (
        "Duplicate parameters found in params and keyword arguments: ['max_tokens']"
        in str(e.value)
    )
