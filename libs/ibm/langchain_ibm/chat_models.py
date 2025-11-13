"""IBM watsonx.ai chat wrapper."""

from __future__ import annotations

import ast
import contextlib
import json
import logging
import warnings
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

from ibm_watsonx_ai import APIClient  # type: ignore[import-untyped]
from ibm_watsonx_ai.foundation_models import (  # type: ignore[import-untyped]
    ModelInference,
)
from ibm_watsonx_ai.foundation_models.schema import (  # type: ignore[import-untyped]
    BaseSchema,
    TextChatParameters,
)
from ibm_watsonx_ai.gateway import Gateway  # type: ignore[import-untyped]
from langchain_core.language_models.chat_models import (  # type: ignore[attr-defined]
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.utils._merge import merge_dicts
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import (
    TypeBaseModel,
    is_basemodel_subclass,
)
from langchain_core.utils.utils import secret_from_env
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import Self, override

from langchain_ibm.utils import (
    async_gateway_error_handler,
    check_duplicate_chat_params,
    extract_chat_params,
    gateway_error_handler,
    normalize_api_key,
    resolve_watsonx_credentials,
    secret_from_env_multi,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence

    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def normalize_tool_arguments(args_str: str) -> str:
    """Ensure arguments is always a proper JSON string.

    Handles:
    - JSON string
    - Python dict string
    - Extra wrapping quotes like '"{...}"'
    Args:
        args_str: tool call args_str.

    Returns:
        The LangChain tool call arguments args_str.
    """
    # Try to parse as JSON
    try:
        parsed = json.loads(args_str)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(parsed, str):
            json.loads(parsed)
            return parsed
        return args_str

    # Try Python literal (e.g., "{'a': 1}")
    obj: Any = ast.literal_eval(args_str)
    return json.dumps(obj, ensure_ascii=False)


def _convert_dict_to_message(_dict: Mapping[str, Any], call_id: str) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.
        call_id: call id

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    name = _dict.get("name")
    id_ = call_id
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), id=id_, name=name)
    if role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: dict[str, Any] = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            for raw_tool_call in raw_tool_calls:
                ## Code change to support langgraph with A2A and graph.astream.
                if "function" in raw_tool_call:
                    func = raw_tool_call.get("function", {})
                    if "arguments" in func:
                        raw_args = raw_tool_call["function"]["arguments"]
                        json_args_str = normalize_tool_arguments(raw_args)
                        raw_tool_call["function"]["arguments"] = json_args_str

                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e)),
                    )
        if audio := _dict.get("audio"):
            additional_kwargs["audio"] = audio
        if reasoning_content := _dict.get("reasoning_content"):
            additional_kwargs["reasoning_content"] = reasoning_content
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""), name=name, id=id_)
    if role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""),
            name=cast("str", _dict.get("name")),
            id=id_,
        )
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=cast("str", _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )
    return ChatMessage(content=_dict.get("content", ""), role=role, id=id_)  # type: ignore[unused-ignore]


def _format_message_content(content: Any) -> Any:
    """Format message content."""
    if content and isinstance(content, list):
        formatted_content = []
        for block in content:
            # Remove unexpected block types
            if (
                isinstance(block, dict)
                and "type" in block
                and block["type"] in {"tool_use", "thinking", "reasoning_content"}
            ):
                continue

            # Image blocks
            if isinstance(block, dict) and block.get("type") == "image":
                if (data := block.get("base64")) and (
                    mime_type := block.get("mime_type")
                ):
                    formatted_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{data}"},
                        }
                    )
                else:
                    continue
            else:
                formatted_content.append(block)
    else:
        formatted_content = content

    return formatted_content


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: dict[str, Any] = {"content": _format_message_content(message.content)}
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_watsonx_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_watsonx_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            tool_call_supported_props = {"id", "type", "function"}
            message_dict["tool_calls"] = [
                {k: v for k, v in tool_call.items() if k in tool_call_supported_props}
                for tool_call in message_dict["tool_calls"]
            ]
        elif "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None

        audio: dict[str, Any] | None = None
        for block in message.content:
            if (
                isinstance(block, dict)
                and block.get("type") == "audio"
                and (id_ := block.get("id"))
            ):
                audio = {"id": id_}
        if not audio and "audio" in message.additional_kwargs:
            raw_audio = message.additional_kwargs["audio"]
            audio = (
                {"id": message.additional_kwargs["audio"]["id"]}
                if "id" in raw_audio
                else raw_audio
            )
        if audio:
            message_dict["audio"] = audio
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id

        supported_props = {"content", "role", "tool_call_id"}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        error_msg = f"Got unknown type {message}"
        raise TypeError(error_msg)
    return message_dict


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any],
    default_class: type[BaseMessageChunk],
    call_id: str,
    *,
    is_first_tool_chunk: bool,
) -> BaseMessageChunk:
    id_ = call_id
    role = cast("str", _dict.get("role"))
    content = cast("str", _dict.get("content") or "")
    additional_kwargs: dict[str, Any] = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        with contextlib.suppress(KeyError):
            tool_call_chunks = [
                tool_call_chunk(
                    name=rtc["function"].get("name")
                    if is_first_tool_chunk or (rtc.get("id") is not None)
                    else None,
                    args=rtc["function"].get("arguments"),
                    # `id` is provided only for the first delta with unique tool_calls
                    # (multiple tool calls scenario)
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]

    if reasoning_content := _dict.get("reasoning_content"):
        additional_kwargs["reasoning_content"] = reasoning_content

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,  # type: ignore[unused-ignore]
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content, id=id_)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content,
            tool_call_id=_dict["tool_call_id"],
            id=id_,
        )
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    return default_class(content=content, id=id_)  # type: ignore[call-arg]


def _convert_chunk_to_generation_chunk(
    chunk: dict[str, Any],
    default_chunk_class: type,
    *,
    is_first_tool_chunk: bool,
    _prompt_tokens_included: bool,
) -> ChatGenerationChunk | None:
    token_usage = chunk.get("usage")
    choices = chunk.get("choices", [])

    usage_metadata: UsageMetadata | None = (
        _create_usage_metadata(
            token_usage, _prompt_tokens_included=_prompt_tokens_included
        )
        if token_usage
        else None
    )

    if len(choices) == 0:
        # logprobs is implicitly None
        return ChatGenerationChunk(
            message=default_chunk_class(content="", usage_metadata=usage_metadata),
        )

    choice = choices[0]
    if choice["delta"] is None:
        return None

    message_chunk = _convert_delta_to_message_chunk(
        choice["delta"],
        default_chunk_class,
        chunk["id"],
        is_first_tool_chunk=is_first_tool_chunk,
    )
    generation_info = {}

    if finish_reason := choice.get("finish_reason"):
        generation_info["finish_reason"] = finish_reason
        if model_name := chunk.get("model"):
            generation_info["model_name"] = model_name
        if system_fingerprint := chunk.get("system_fingerprint"):
            generation_info["system_fingerprint"] = system_fingerprint

    logprobs = choice.get("logprobs")
    if logprobs:
        generation_info["logprobs"] = logprobs

    if usage_metadata and isinstance(message_chunk, AIMessageChunk):
        message_chunk.usage_metadata = usage_metadata

    return ChatGenerationChunk(
        message=message_chunk,
        generation_info=generation_info or None,
    )


class ChatWatsonx(BaseChatModel):
    r"""`IBM watsonx.ai` chat models integration.

    ???+ info "Setup"

        To use, you should have `langchain_ibm` python package installed,
        and the environment variable `WATSONX_API_KEY` set with your API key, or pass
        it as a named parameter `api_key` to the constructor.

        ```bash
        pip install -U langchain-ibm

        # or using uv
        uv add langchain-ibm
        ```

        ```bash
        export WATSONX_API_KEY="your-api-key"
        ```

        !!! deprecated
            `apikey` and `WATSONX_APIKEY` are deprecated and will be removed in
            version `2.0.0`. Use `api_key` and `WATSONX_API_KEY` instead.

    ??? info "Instantiate"

        Create a model instance with desired params. For example:

        ```python
        from langchain_ibm import ChatWatsonx
        from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

        parameters = TextChatParameters(
            top_p=1, temperature=0.5, max_completion_tokens=None
        )

        model = ChatWatsonx(
            model_id="meta-llama/llama-3-3-70b-instruct",
            url="https://us-south.ml.cloud.ibm.com",
            project_id="*****",
            params=parameters,
            # api_key="*****"
        )
        ```

    ??? info "Invoke"

        Generate a response from the model:

        ```python
        messages = [
            (
                "system",
                "You are a helpful translator. Translate the user sentence to French.",
            ),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

        Results in an `AIMessage` response:

        ```python
        AIMessage(
            content="J'adore programmer.",
            additional_kwargs={},
            response_metadata={
                "token_usage": {
                    "completion_tokens": 7,
                    "prompt_tokens": 30,
                    "total_tokens": 37,
                },
                "model_name": "ibm/granite-3-3-8b-instruct",
                "system_fingerprint": "",
                "finish_reason": "stop",
            },
            id="chatcmpl-529352c4-93ba-4801-8f1d-a3b4e3935194---daed91fb74d0405f200db1e63da9a48a---7a3ef799-4413-47e4-b24c-85d267e37fa2",
            usage_metadata={"input_tokens": 30, "output_tokens": 7, "total_tokens": 37},
        )
        ```

    ??? info "Stream"

        Stream a response from the model:

        ```python
        for chunk in model.stream(messages):
            print(chunk.text)
        ```

        Results in a sequence of `AIMessageChunk` objects with partial content:

        ```python
        AIMessageChunk(content="", id="run--e48a38d3-1500-4b5e-870c-6313e8cff775")
        AIMessageChunk(content="J", id="run--e48a38d3-1500-4b5e-870c-6313e8cff775")
        AIMessageChunk(content="'", id="run--e48a38d3-1500-4b5e-870c-6313e8cff775")
        AIMessageChunk(content="ad", id="run--e48a38d3-1500-4b5e-870c-6313e8cff775")
        AIMessageChunk(content="or", id="run--e48a38d3-1500-4b5e-870c-6313e8cff775")
        AIMessageChunk(
            content=" programmer", id="run--e48a38d3-1500-4b5e-870c-6313e8cff775"
        )
        AIMessageChunk(content=".", id="run--e48a38d3-1500-4b5e-870c-6313e8cff775")
        AIMessageChunk(
            content="",
            response_metadata={
                "finish_reason": "stop",
                "model_name": "ibm/granite-3-3-8b-instruct",
            },
            id="run--e48a38d3-1500-4b5e-870c-6313e8cff775",
        )
        AIMessageChunk(
            content="",
            id="run--e48a38d3-1500-4b5e-870c-6313e8cff775",
            usage_metadata={"input_tokens": 30, "output_tokens": 7, "total_tokens": 37},
        )
        ```

        To collect the full message, you can concatenate the chunks:

        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk

        full
        ```

        ```python
        AIMessageChunk(
            content="J'adore programmer.",
            response_metadata={
                "finish_reason": "stop",
                "model_name": "ibm/granite-3-3-8b-instruct",
            },
            id="chatcmpl-88a48b71-c149-4a0c-9c02-d6b97ca5dc6c---b7ba15879a8c5283b1e8a3b8db0229f0---0037ca4f-8a74-4f84-a46c-ab3fd1294f24",
            usage_metadata={"input_tokens": 30, "output_tokens": 7, "total_tokens": 37},
        )
        ```

    ??? info "Async"

        Asynchronous equivalents of `invoke`, `stream`, and `batch` are also available:

        ```python
        # Invoke
        await model.ainvoke(messages)

        # Stream
        async for chunk in model.astream(messages):
            print(chunk.text)

        # Batch
        await model.abatch([messages])
        ```

        Results in an `AIMessage` response:

        ```python
        AIMessage(
            content="J'adore programmer.",
            additional_kwargs={},
            response_metadata={
                "token_usage": {
                    "completion_tokens": 7,
                    "prompt_tokens": 30,
                    "total_tokens": 37,
                },
                "model_name": "ibm/granite-3-3-8b-instruct",
                "system_fingerprint": "",
                "finish_reason": "stop",
            },
            id="chatcmpl-5bef2d81-ef56-463b-a8fa-c2bcc2a3c348---821e7750d18925f2b36226db66667e26---6396c786-9da9-4468-883e-11ed90a05937",
            usage_metadata={"input_tokens": 30, "output_tokens": 7, "total_tokens": 37},
        )
        ```

        For batched calls, results in a `list[AIMessage]`.

    ??? info "Tool calling"

        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


        model_with_tools = model.bind_tools(
            [GetWeather, GetPopulation]
            # strict = True  # Enforce tool args schema is respected
        )
        ai_msg = model_with_tools.invoke(
            "Which city is hotter today and which is bigger: LA or NY?"
        )
        ai_msg.tool_calls
        ```

        ```python
        [
            {
                "name": "GetWeather",
                "args": {"location": "Los Angeles, CA"},
                "id": "chatcmpl-tool-59632abcee8f48a18a5f3a81422b917b",
                "type": "tool_call",
            },
            {
                "name": "GetWeather",
                "args": {"location": "New York, NY"},
                "id": "chatcmpl-tool-c6f3b033b4594918bb53f656525b0979",
                "type": "tool_call",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "Los Angeles, CA"},
                "id": "chatcmpl-tool-175a23281e4747ea81cbe472b8e47012",
                "type": "tool_call",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "New York, NY"},
                "id": "chatcmpl-tool-e1ccc534835945aebab708eb5e685bf7",
                "type": "tool_call",
            },
        ]
        ```

    ??? info "Reasoning output"

        ```python
        from langchain_ibm import ChatWatsonx
        from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

        parameters = TextChatParameters(
            include_reasoning=True, reasoning_effort="medium"
        )

        model = ChatWatsonx(
            model_id="openai/gpt-oss-120b",
            url="https://us-south.ml.cloud.ibm.com",
            project_id="*****",
            params=parameters,
            # api_key="*****"
        )

        response = model.invoke("What is 3^3?")

        # Response text
        print(f"Output: {response.content}")

        # Reasoning summaries
        print(f"Reasoning: {response.additional_kwargs['reasoning_content']}")
        ```

        ```txt
        Output: 3^3 = 27
        Reasoning: The user asks "What is 3^3?" That's 27. Provide answer.
        ```

        !!! version-added "Added in version 0.3.19: Updated `AIMessage` format"
            [`langchain-ibm >= 0.3.19`](https://pypi.org/project/langchain-ibm/#history)
            allows users to set Reasoning output parameters and will format output from
            reasoning summaries into `additional_kwargs` field.

    ??? info "Structured output"

        ```python
        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: int | None = Field(description="How funny the joke is, 1 to 10")


        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")
        ```

        ```python
        Joke(
            setup="Why was the cat sitting on the computer?",
            punchline="To keep an eye on the mouse!",
            rating=None,
        )
        ```

        See `with_structured_output` for more info.

    ??? info "JSON mode"

        ```python
        json_model = model.bind(response_format={"type": "json_object"})
        ai_msg = json_model.invoke(
            “Return JSON with 'random_ints': an array of 10 random integers from 0-99.”
        )
        ai_msg.content
        ```

        ```txt
        '{\n  "random_ints": [12, 34, 56, 78, 10, 22, 44, 66, 88, 99]\n}'
        ```

    ??? info "Image input"

        ```python
        import base64
        import httpx
        from langchain.messages import HumanMessage

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )

        ai_msg = model.invoke([message])
        ai_msg.content
        ```

        ```txt
        "The weather in the image presents a clear, sunny day with good visibility
        and no immediate signs of rain or strong winds. The vibrant blue sky with
        scattered white clouds gives the impression of a tranquil, pleasant day
        conducive to outdoor activities."
        ```

    ??? info "Token usage"

        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```

        ```txt
        {'input_tokens': 30, 'output_tokens': 7, 'total_tokens': 37}
        ```

        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full.usage_metadata
        ```

        ```txt
        {'input_tokens': 30, 'output_tokens': 7, 'total_tokens': 37}
        ```

    ??? info "Logprobs"

        ```python
        logprobs_model = model.bind(logprobs=True)
        ai_msg = logprobs_model.invoke(messages)
        ai_msg.response_metadata["logprobs"]
        ```

        ```txt
        {
            'content': [
                {
                    'token': 'J',
                    'logprob': -0.0017940393
                },
                {
                    'token': "'",
                    'logprob': -1.7523613e-05
                },
                {
                    'token': 'ad',
                    'logprob': -0.16112353
                },
                {
                    'token': 'ore',
                    'logprob': -0.0003091811
                },
                {
                    'token': ' programmer',
                    'logprob': -0.24849245
                },
                {
                    'token': '.',
                    'logprob': -2.5033638e-05
                },
                {
                    'token': '<|end_of_text|>',
                    'logprob': -7.080781e-05
                }
            ]
        }
        ```

    ??? info "Response metadata"

        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```

        ```txt
        {
            'token_usage': {
                'completion_tokens': 7,
                'prompt_tokens': 30,
                'total_tokens': 37
            },
            'model_name': 'ibm/granite-3-3-8b-instruct',
            'system_fingerprint': '',
            'finish_reason': 'stop'
        }
        ```
    """

    model_id: str | None = None
    """Type of model to use."""

    model: str | None = None
    """
    Name or alias of the foundation model to use.
    When using IBM's watsonx.ai Model Gateway (public preview), you can specify any
    supported third-party model—OpenAI, Anthropic, NVIDIA, Cerebras, or IBM's own
    Granite series—via a single, OpenAI-compatible interface. Models must be explicitly
    provisioned (opt-in) through the Gateway to ensure secure, vendor-agnostic access
    and easy switch-over without reconfiguration.

    For more details on configuration and usage, see [IBM watsonx Model Gateway docs](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-model-gateway.html?context=wx&audience=wdp)
    """

    deployment_id: str | None = None
    """Type of deployed model to use."""

    project_id: str | None = None
    """ID of the Watson Studio project."""

    space_id: str | None = None
    """ID of the Watson Studio space."""

    url: SecretStr = Field(
        alias="url",
        default_factory=secret_from_env("WATSONX_URL", default=None),  # type: ignore[assignment]
    )
    """URL to the Watson Machine Learning or CPD instance."""

    apikey: SecretStr | None = None
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env_multi(
            names_priority=["WATSONX_API_KEY", "WATSONX_APIKEY"],
            deprecated={"WATSONX_APIKEY"},
        ),
        serialization_alias="api_key",
        validation_alias=AliasChoices("api_key", "apikey"),  # accept both on input
        description="API key to the Watson Machine Learning or CPD instance.",
    )
    """API key to the Watson Machine Learning or CPD instance."""

    token: SecretStr | None = Field(
        alias="token",
        default_factory=secret_from_env("WATSONX_TOKEN", default=None),
    )
    """Token to the CPD instance."""

    password: SecretStr | None = Field(
        alias="password",
        default_factory=secret_from_env("WATSONX_PASSWORD", default=None),
    )
    """Password to the CPD instance."""

    username: SecretStr | None = Field(
        alias="username",
        default_factory=secret_from_env("WATSONX_USERNAME", default=None),
    )
    """Username to the CPD instance."""

    instance_id: SecretStr | None = Field(
        alias="instance_id",
        default_factory=secret_from_env("WATSONX_INSTANCE_ID", default=None),
    )
    """Instance_id of the CPD instance."""

    version: SecretStr | None = None
    """Version of the CPD instance."""

    params: dict[str, Any] | TextChatParameters | None = None
    """Model parameters to use during request generation.

    !!! note
        `ValueError` is raised if the same Chat generation parameter is provided
        within the params attribute and as keyword argument.
    """

    frequency_penalty: float | None = None
    """Positive values penalize new tokens based on their existing frequency in the
    text so far, decreasing the model's likelihood to repeat the same line verbatim."""

    logprobs: bool | None = None
    """Whether to return log probabilities of the output tokens or not.
    If true, returns the log probabilities of each output token returned
    in the content of message."""

    top_logprobs: int | None = None
    """An integer specifying the number of most likely tokens to return at each
    token position, each with an associated log probability. The option logprobs
    must be set to true if this parameter is used."""

    max_tokens: int | None = None
    """The maximum number of tokens that can be generated in the chat completion.
    The total length of input tokens and generated tokens is limited by the
    model's context length.
    This value is now deprecated in favor of 'max_completion_tokens' parameter."""

    max_completion_tokens: int | None = None
    """The maximum number of tokens that can be generated in the chat completion.
    The total length of input tokens and generated tokens is limited by the
    model's context length."""

    n: int | None = None
    """How many chat completion choices to generate for each input message.
    Note that you will be charged based on the number of generated tokens across
    all of the choices. Keep n as 1 to minimize costs."""

    presence_penalty: float | None = None
    """Positive values penalize new tokens based on whether they appear in the
    text so far, increasing the model's likelihood to talk about new topics."""

    temperature: float | None = None
    """What sampling temperature to use. Higher values like 0.8 will make the
    output more random, while lower values like 0.2 will make it more focused
    and deterministic.

    We generally recommend altering this or top_p but not both."""

    response_format: dict[str, Any] | None = None
    """The chat response format parameters."""

    top_p: float | None = None
    """An alternative to sampling with temperature, called nucleus sampling,
    where the model considers the results of the tokens with top_p probability
    mass. So 0.1 means only the tokens comprising the top 10% probability mass
    are considered.

    We generally recommend altering this or temperature but not both."""

    time_limit: int | None = None
    """Time limit in milliseconds - if not completed within this time,
    generation will stop."""

    logit_bias: dict[str, int] | None = None
    """Increasing or decreasing probability of tokens being selected
    during generation."""

    seed: int | None = None
    """Random number generator seed to use in sampling mode
    for experimental repeatability."""

    stop: list[str] | None = None
    """Stop sequences are one or more strings which will cause the text generation
    to stop if/when they are produced as part of the output."""

    chat_template_kwargs: dict[str, Any] | None = None
    """Additional chat template parameters."""

    verify: str | bool | None = None
    """You can pass one of following as verify:
        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * True - default path to truststore will be taken
        * False - no verification will be made"""

    validate_model: bool = True
    """Model ID validation."""

    streaming: bool = False
    """Whether to stream the results or not."""

    watsonx_model: ModelInference = Field(default=None, exclude=True)  #: :meta private:

    watsonx_model_gateway: Gateway = Field(
        default=None,
        exclude=True,
    )  #: :meta private:

    watsonx_client: APIClient | None = Field(default=None, exclude=True)

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Is lc serializable."""
        return False

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "watsonx-chat"

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "ibm"
        if self.model_id:
            params["ls_model_name"] = self.model_id
        return params

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Mapping of secret environment variables."""
        return {
            "url": "WATSONX_URL",
            "api_key": "WATSONX_API_KEY",  # preferred
            "apikey": "WATSONX_APIKEY",
            "token": "WATSONX_TOKEN",
            "password": "WATSONX_PASSWORD",
            "username": "WATSONX_USERNAME",
            "instance_id": "WATSONX_INSTANCE_ID",
        }

    @model_validator(mode="before")
    @classmethod
    def _normalize_and_warn_deprecated_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Handle deprecated input kwarg name `apikey` vs new `api_key`.
            data = normalize_api_key(data=data)
        return data

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that credentials and python package exists in environment."""
        self.params = self.params or {}

        if isinstance(self.params, BaseSchema):
            self.params = self.params.to_dict()

        check_duplicate_chat_params(self.params, self.__dict__)

        self.params.update(
            {
                k: v
                for k, v in {
                    param: getattr(self, param)
                    for param in ChatWatsonx._get_supported_chat_params()
                }.items()
                if v is not None
            },
        )
        if self.watsonx_model_gateway is not None:
            error_msg = (
                "Passing the 'watsonx_model_gateway' parameter to the ChatWatsonx "
                "constructor is not supported yet.",
            )
            raise NotImplementedError(error_msg)

        if isinstance(self.watsonx_model, ModelInference):
            self.model_id = self.watsonx_model.model_id
            self.deployment_id = getattr(self.watsonx_model, "deployment_id", "")
            self.project_id = self.watsonx_model._client.default_project_id  # noqa: SLF001
            self.space_id = self.watsonx_model._client.default_space_id  # noqa: SLF001
            self.params = self.watsonx_model.params
            self.watsonx_client = self.watsonx_model._client  # noqa: SLF001

        elif isinstance(self.watsonx_client, APIClient):
            if sum(map(bool, (self.model, self.model_id, self.deployment_id))) != 1:
                error_msg = (
                    "The parameters 'model', 'model_id' and 'deployment_id' are "
                    "mutually exclusive. Please specify exactly one of these "
                    "parameters when initializing ChatWatsonx.",
                )
                raise ValueError(error_msg)
            if self.model is not None:
                watsonx_model_gateway = Gateway(
                    api_client=self.watsonx_client,
                    verify=self.verify,
                )
                self.watsonx_model_gateway = watsonx_model_gateway
            else:
                watsonx_model = ModelInference(
                    model_id=self.model_id,
                    deployment_id=self.deployment_id,
                    params=self.params,
                    api_client=self.watsonx_client,
                    project_id=self.project_id,
                    space_id=self.space_id,
                    verify=self.verify,
                    validate=self.validate_model,
                )
                self.watsonx_model = watsonx_model
        else:
            if sum(map(bool, (self.model, self.model_id, self.deployment_id))) != 1:
                error_msg = (
                    "The parameters 'model', 'model_id' and 'deployment_id' are "
                    "mutually exclusive. Please specify exactly one of these "
                    "parameters when initializing ChatWatsonx.",
                )
                raise ValueError(error_msg)

            credentials = resolve_watsonx_credentials(
                url=self.url,
                api_key=self.api_key,
                token=self.token,
                password=self.password,
                username=self.username,
                instance_id=self.instance_id,
                version=self.version,
                verify=self.verify,
            )
            if self.model is not None:
                watsonx_model_gateway = Gateway(
                    credentials=credentials,
                    verify=self.verify,
                )
                self.watsonx_model_gateway = watsonx_model_gateway
            else:
                watsonx_model = ModelInference(
                    model_id=self.model_id,
                    deployment_id=self.deployment_id,
                    credentials=credentials,
                    params=self.params,
                    project_id=self.project_id,
                    space_id=self.space_id,
                    verify=self.verify,
                    validate=self.validate_model,
                )
                self.watsonx_model = watsonx_model

        return self

    @gateway_error_handler
    def _call_model_gateway(
        self, *, model: str, messages: list[dict[str, Any]], **params: Any
    ) -> Any:
        return self.watsonx_model_gateway.chat.completions.create(
            model=model,
            messages=messages,
            **params,
        )

    @async_gateway_error_handler
    async def _acall_model_gateway(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **params: Any,
    ) -> Any:
        return await self.watsonx_model_gateway.chat.completions.acreate(
            model=model,
            messages=messages,
            **params,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop, **kwargs)
        updated_params = self._merge_params(params, kwargs)
        if self.watsonx_model_gateway is not None:
            call_kwargs = {**kwargs, **updated_params}
            response = self._call_model_gateway(
                model=self.model,
                messages=message_dicts,
                **call_kwargs,
            )
        else:
            response = self.watsonx_model.chat(
                messages=message_dicts,
                **(kwargs | {"params": updated_params}),
            )
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop, **kwargs)
        updated_params = self._merge_params(params, kwargs)
        if self.watsonx_model_gateway is not None:
            call_kwargs = {**kwargs, **updated_params}
            response = await self._acall_model_gateway(
                model=self.model,
                messages=message_dicts,
                **call_kwargs,
            )
        else:
            response = await self.watsonx_model.achat(
                messages=message_dicts,
                **(kwargs | {"params": updated_params}),
            )
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop, **kwargs)
        updated_params = self._merge_params(params, kwargs)

        if self.watsonx_model_gateway is not None:
            call_kwargs = {**kwargs, **updated_params, "stream": True}
            chunk_iter = self._call_model_gateway(
                model=self.model,
                messages=message_dicts,
                **call_kwargs,
            )
        else:
            call_kwargs = {**kwargs, "params": updated_params}
            chunk_iter = self.watsonx_model.chat_stream(
                messages=message_dicts,
                **call_kwargs,
            )

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        is_first_tool_chunk = True
        _prompt_tokens_included = False

        for chunk in chunk_iter:
            chunk_data = chunk if isinstance(chunk, dict) else chunk.model_dump()
            generation_chunk = _convert_chunk_to_generation_chunk(
                chunk_data,
                default_chunk_class,
                is_first_tool_chunk=is_first_tool_chunk,
                _prompt_tokens_included=_prompt_tokens_included,
            )
            if generation_chunk is None:
                continue

            if (
                hasattr(generation_chunk.message, "usage_metadata")
                and generation_chunk.message.usage_metadata
            ):
                _prompt_tokens_included = True
            default_chunk_class = generation_chunk.message.__class__
            logprobs = (generation_chunk.generation_info or {}).get("logprobs")
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            if hasattr(generation_chunk.message, "tool_calls") and isinstance(
                generation_chunk.message.tool_calls,
                list,
            ):
                first_tool_call = (
                    generation_chunk.message.tool_calls[0]
                    if generation_chunk.message.tool_calls
                    else None
                )
                if isinstance(first_tool_call, dict) and first_tool_call.get("name"):
                    is_first_tool_chunk = False

            yield generation_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop, **kwargs)
        updated_params = self._merge_params(params, kwargs)

        if self.watsonx_model_gateway is not None:
            call_kwargs = {**kwargs, **updated_params, "stream": True}
            chunk_iter = await self._acall_model_gateway(
                model=self.model,
                messages=message_dicts,
                **call_kwargs,
            )
        else:
            call_kwargs = {**kwargs, "params": updated_params}
            chunk_iter = await self.watsonx_model.achat_stream(
                messages=message_dicts,
                **call_kwargs,
            )

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        is_first_tool_chunk = True
        _prompt_tokens_included = False

        async for chunk in chunk_iter:
            chunk_data = chunk if isinstance(chunk, dict) else chunk.model_dump()
            generation_chunk = _convert_chunk_to_generation_chunk(
                chunk_data,
                default_chunk_class,
                is_first_tool_chunk=is_first_tool_chunk,
                _prompt_tokens_included=_prompt_tokens_included,
            )
            if generation_chunk is None:
                continue

            if (
                hasattr(generation_chunk.message, "usage_metadata")
                and generation_chunk.message.usage_metadata
            ):
                _prompt_tokens_included = True
            default_chunk_class = generation_chunk.message.__class__
            logprobs = (generation_chunk.generation_info or {}).get("logprobs")
            if run_manager:
                await run_manager.on_llm_new_token(
                    generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            if hasattr(generation_chunk.message, "tool_calls") and isinstance(
                generation_chunk.message.tool_calls,
                list,
            ):
                first_tool_call = (
                    generation_chunk.message.tool_calls[0]
                    if generation_chunk.message.tool_calls
                    else None
                )
                if isinstance(first_tool_call, dict) and first_tool_call.get("name"):
                    is_first_tool_chunk = False

            yield generation_chunk

    @staticmethod
    def _merge_params(params: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        param_updates = {}
        for k in ChatWatsonx._get_supported_chat_params():
            if kwargs.get(k) is not None:
                param_updates[k] = kwargs.pop(k)

        if kwargs.get("params"):
            merged_params = merge_dicts(params, param_updates)
        else:
            merged_params = params | param_updates

        return merged_params

    def _create_message_dicts(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = extract_chat_params(kwargs, self.params)

        if stop is not None:
            if params and "stop_sequences" in params:
                error_msg = (
                    "`stop_sequences` found in both the input and default params."
                )
                raise ValueError(error_msg)
            params = (params or {}) | {"stop_sequences": stop}
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params or {}

    def _create_chat_result(
        self,
        response: dict[str, Any],
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        generations = []

        if response.get("error"):
            raise ValueError(response.get("error"))

        token_usage = response.get("usage", {})

        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"], response["id"])

            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(
                    token_usage, _prompt_tokens_included=False
                )
            generation_info = generation_info or {}
            generation_info["finish_reason"] = (
                res.get("finish_reason")
                if res.get("finish_reason") is not None
                else generation_info.get("finish_reason")
            )
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_name": response.get("model_id", self.model_id),
            "system_fingerprint": response.get("system_fingerprint", ""),
        }

        return ChatResult(generations=generations, llm_output=llm_output)

    @staticmethod
    def _get_supported_chat_params() -> list[str]:
        # watsonx.ai Chat API doc: https://cloud.ibm.com/apidocs/watsonx-ai#text-chat
        return [
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "presence_penalty",
            "response_format",
            "temperature",
            "top_p",
            "time_limit",
            "logit_bias",
            "seed",
            "stop",
            "chat_template_kwargs",
        ]

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: dict[str, Any] | str | bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call. Options are:

                - `str` of the form `'<<tool_name>>'`: calls `<<tool_name>>` tool.
                - `'auto'`: automatically selects a tool (including no tool).
                - `'none'`: does not call a tool.
                - `'any'` or `'required'` or `True`: force at least one tool to be
                  called.
                - `dict` of the form
                  `{"type": "function", "function": {"name": <<tool_name>>}}`:
                  calls `<<tool_name>>` tool.
                - `False` or `None`: no effect, default OpenAI behavior.

            strict: If `True`, model output is guaranteed to exactly match the JSON
                Schema provided in the tool definition.
                The input schema will also be validated according to the supported
                schemas.
                If `False`, input schema will not be validated and model output will not
                be validated.
                If `None`, `strict` argument will not be passed to the model.

            kwargs: Any additional parameters are passed directly to `bind`.
        """
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "any", "required"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                # We support 'any' since other models use this instead of 'required'.
                if tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                tool_names = [
                    formatted_tool["function"]["name"]
                    for formatted_tool in formatted_tools
                ]
                if not any(
                    tool_name == tool_choice["function"]["name"]
                    for tool_name in tool_names
                ):
                    error_msg = (
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}.",
                    )
                    raise ValueError(error_msg)
            else:
                error_msg = (  # type: ignore[unreachable]
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}",
                )
                raise ValueError(error_msg)

            if isinstance(tool_choice, str):
                kwargs["tool_choice_option"] = tool_choice
            else:
                kwargs["tool_choice"] = tool_choice
        else:
            kwargs["tool_choice_option"] = "auto"

        return super().bind(tools=formatted_tools, **kwargs)

    @override
    def with_structured_output(
        self,
        schema: dict[str, Any] | type | None = None,
        *,
        method: Literal[
            "function_calling",
            "json_mode",
            "json_schema",
        ] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        r"""Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

                - a JSON Schema,
                - a TypedDict class,
                - or a Pydantic class,

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

            method: The method for steering model generation, one of:

                - `'function_calling'`: uses tool-calling features.
                - `'json_schema'`: uses dedicated structured output features.
                - `'json_mode'`: uses JSON mode.

            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys `'raw'`, `'parsed'`, and `'parsing_error'`.

            strict:

                - `True`:
                    Model output is guaranteed to exactly match the schema.
                    The input schema will also be validated according to the
                    supported schemas.
                - `False`:
                    Input schema will not be validated and model output will not be
                    validated.
                - `None`:
                    `strict` argument will not be passed to the model.

            kwargs: Additional keyword args


        Returns:
            A Runnable that takes same inputs as a `langchain_core.language_models.chat.BaseChatModel`.

            If `include_raw` is True, then Runnable outputs a dict with keys:

            - `'raw'`: BaseMessage
            - `'parsed'`: None if there was a parsing error, otherwise the type depends on the `schema` as described above.
            - `'parsing_error'`: Optional[BaseException]

        ??? note "Example: `schema=Pydantic` class, `method='function_calling'`, `include_raw=True`"

            ```python
            from langchain_ibm import ChatWatsonx
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            model = ChatWatsonx(...)
            structured_model = model.with_structured_output(
                AnswerWithJustification, include_raw=True
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "call_Ao02pnFYXD6GN1yzc0uXPsvF",
                                "function": {
                                    "arguments": '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}',
                                    "name": "AnswerWithJustification",
                                },
                                "type": "function",
                            }
                        ]
                    },
                ),
                "parsed": AnswerWithJustification(
                    answer="They weigh the same.",
                    justification="Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.",
                ),
                "parsing_error": None,
            }
            ```

        ??? note "Example: `schema=JSON` schema, `method='function_calling'`, `include_raw=False`"

            ```python
            from langchain_ibm import ChatWatsonx
            from pydantic import BaseModel
            from langchain_core.utils.function_calling import convert_to_openai_tool


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            dict_schema = convert_to_openai_tool(AnswerWithJustification)
            model = ChatWatsonx(...)
            structured_model = model.with_structured_output(dict_schema)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "answer": "They weigh the same",
                "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.",
            }
            ```

        ??? note "Example: `schema=Pydantic` class, `method='json_schema'`, `include_raw=True`"

            ```python
            from langchain_ibm import ChatWatsonx
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            model = ChatWatsonx(...)
            structured_model = model.with_structured_output(
                AnswerWithJustification, method="json_schema", include_raw=True
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "chatcmpl-tool-bfbd6f6dd33b438990c5ddf277485971",
                                "type": "function",
                                "function": {
                                    "name": "AnswerWithJustification",
                                    "arguments": '{"answer": "They weigh the same", "justification": "A pound is a unit of weight or mass, so both a pound of bricks and a pound of feathers weigh the same amount, one pound."}',
                                },
                            }
                        ]
                    },
                    response_metadata={
                        "token_usage": {
                            "completion_tokens": 45,
                            "prompt_tokens": 275,
                            "total_tokens": 320,
                        },
                        "model_name": "meta-llama/llama-3-3-70b-instruct",
                        "system_fingerprint": "",
                        "finish_reason": "stop",
                    },
                    id="chatcmpl-461ca5bd-1982-412c-b886-017c483bf481---8c18b06eead65ae4691364798787bda7---71896588-efa5-439f-a25f-d1abfe289f5a",
                    tool_calls=[
                        {
                            "name": "AnswerWithJustification",
                            "args": {
                                "answer": "They weigh the same",
                                "justification": "A pound is a unit of weight or mass, so both a pound of bricks and a pound of feathers weigh the same amount, one pound.",
                            },
                            "id": "chatcmpl-tool-bfbd6f6dd33b438990c5ddf277485971",
                            "type": "tool_call",
                        }
                    ],
                    usage_metadata={
                        "input_tokens": 275,
                        "output_tokens": 45,
                        "total_tokens": 320,
                    },
                ),
                "parsed": AnswerWithJustification(
                    answer="They weigh the same",
                    justification="A pound is a unit of weight or mass, so both a pound of bricks and a pound of feathers weigh the same amount, one pound.",
                ),
                "parsing_error": None,
            }
            ```

        ??? note "Example: `schema=function` schema, `method='json_schema'`, `include_raw=False`"

            ```python
            from langchain_ibm import ChatWatsonx
            from pydantic import BaseModel

            function__schema = {
                "name": "AnswerWithJustification",
                "description": "An answer to the user question along with justification for the answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "justification": {
                            "description": "A justification for the answer.",
                            "type": "string",
                        },
                    },
                    "required": ["answer"],
                },
            }

            model = ChatWatsonx(...)
            structured_model = model.with_structured_output(
                function_schema, method="json_schema", include_raw=False
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "answer": "They weigh the same",
                "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.",
            }
            ```

        ??? note "Example: `schema=Pydantic` class, `method='json_mode'`, `include_raw=True`"

            ```python
            from langchain_ibm import ChatWatsonx
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                answer: str
                justification: str


            model = ChatWatsonx(...)
            structured_model = model.with_structured_output(
                AnswerWithJustification, method="json_mode", include_raw=True
            )

            structured_model.invoke(
                "Answer the following question. "
                "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                "What's heavier a pound of bricks or a pound of feathers?"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'
                ),
                "parsed": AnswerWithJustification(
                    answer="They are both the same weight.",
                    justification="Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.",
                ),
                "parsing_error": None,
            }
            ```

        ??? note "Example: `schema=None`, `method='json_mode'`, `include_raw=True`"

            ```python
            from langchain_ibm import ChatWatsonx

            model = ChatWatsonx(...)
            structured_model = model.with_structured_output(
                method="json_mode", include_raw=True
            )

            structured_model.invoke(
                "Answer the following question. "
                "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                "What's heavier a pound of bricks or a pound of feathers?"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'
                ),
                "parsed": {
                    "answer": "They are both the same weight.",
                    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.",
                },
                "parsing_error": None,
            }
            ```

        """  # noqa: E501
        if strict is not None and method == "json_mode":
            msg = "Argument `strict` is not supported with `method`='json_mode'"
            raise ValueError(msg)

        is_pydantic_schema = _is_pydantic_class(schema)

        if (
            method == "json_schema"
            and is_pydantic_schema
            and issubclass(schema, BaseModelV1)  # type: ignore[arg-type]
        ):
            # Check for Pydantic BaseModel V1
            warnings.warn(
                "Received a Pydantic BaseModel V1 schema. This is not supported by "
                'method="json_schema". Please use method="function_calling" '
                "or specify schema via JSON Schema or Pydantic V2 BaseModel. "
                'Overriding to method="function_calling".',
                stacklevel=2,
            )
            method = "function_calling"

        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                error_msg = (
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
                raise ValueError(error_msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            model = self.bind_tools(
                [schema],
                strict=strict,
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": method, "strict": strict},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                output_parser: Runnable[Any, Any] = PydanticToolsParser(
                    tools=[schema],
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name,
                    first_tool_only=True,
                )
        elif method == "json_mode":
            model = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[unused-ignore]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            if schema is None:
                error_msg = (
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
                raise ValueError(error_msg)
            response_format = _convert_to_openai_response_format(schema, strict=strict)

            bind_kwargs = {
                "response_format": response_format,
                "ls_structured_output_format": {
                    "kwargs": {"method": method, "strict": strict},
                    "schema": convert_to_openai_tool(schema),
                },
            }
            model = self.bind(**bind_kwargs)
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[unused-ignore]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            error_msg = (  # type: ignore[unreachable]
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
            )
            raise ValueError(error_msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser,
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none],
                exception_key="parsing_error",
            )
            return RunnableMap(raw=model) | parser_with_fallback
        return model | output_parser


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _lc_tool_call_to_watsonx_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
        },
    }


def _lc_invalid_tool_call_to_watsonx_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict[str, Any]:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }


def _create_usage_metadata(
    oai_token_usage: dict[str, Any],
    *,
    _prompt_tokens_included: bool,
) -> UsageMetadata:
    input_tokens = (
        oai_token_usage.get("prompt_tokens", 0) if not _prompt_tokens_included else 0
    )
    output_tokens = oai_token_usage.get("completion_tokens", 0)
    total_tokens = oai_token_usage.get("total_tokens", input_tokens + output_tokens)
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _convert_to_openai_response_format(
    schema: dict[str, Any] | type, *, strict: bool | None = None
) -> dict[str, Any] | TypeBaseModel:
    if isinstance(schema, type) and is_basemodel_subclass(schema):
        return schema

    if (
        isinstance(schema, dict)
        and "json_schema" in schema
        and schema.get("type") == "json_schema"
    ):
        response_format = schema
    elif isinstance(schema, dict) and "name" in schema and "schema" in schema:
        response_format = {"type": "json_schema", "json_schema": schema}
    else:
        if strict is None:
            if isinstance(schema, dict) and isinstance(schema.get("strict"), bool):
                strict = schema["strict"]
            else:
                strict = False
        function = convert_to_openai_function(schema, strict=strict)
        function["schema"] = function.pop("parameters")
        response_format = {"type": "json_schema", "json_schema": function}

    if (
        strict is not None
        and strict is not response_format["json_schema"].get("strict")
        and isinstance(schema, dict)
    ):
        msg = (
            f"Output schema already has 'strict' value set to "
            f"{schema['json_schema']['strict']} but 'strict' also passed in to "
            f"with_structured_output as {strict}. Please make sure that "
            f"'strict' is only specified in one place."
        )
        raise ValueError(msg)
    return response_format
