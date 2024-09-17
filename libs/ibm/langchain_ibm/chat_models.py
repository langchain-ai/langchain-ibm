"""IBM watsonx.ai large language chat models wrapper."""

import json
import logging
from datetime import datetime
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)

from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
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
    convert_to_messages,
)
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_ibm.utils import check_for_attribute

logger = logging.getLogger(__name__)


def _convert_dict_to_message(_dict: Mapping[str, Any], call_id: str) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.
        call_id: call id

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("generated_text", ""))
    else:
        additional_kwargs: Dict = {}
        tool_calls = []
        invalid_tool_calls: List[InvalidToolCall] = []
        content = ""

        raw_tool_calls = _dict.get("generated_text", "")

        if "json" in raw_tool_calls:
            try:
                split_raw_tool_calls = raw_tool_calls.split("\n\n")
                for raw_tool_call in split_raw_tool_calls:
                    if "json" in raw_tool_call:
                        json_parts = JsonOutputParser().parse(raw_tool_call)

                        if json_parts["function"]["name"] == "Final Answer":
                            content = json_parts["function"]["arguments"]["output"]
                            break

                        additional_kwargs["tool_calls"] = json_parts

                        parsed = {
                            "name": json_parts["function"]["name"] or "",
                            "args": json_parts["function"]["arguments"] or {},
                            "id": call_id,
                        }
                        tool_calls.append(parsed)

            except:  # noqa: E722
                content = _dict.get("generated_text", "") or ""

        else:
            content = _dict.get("generated_text", "") or ""

        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )


def _format_message_content(content: Any) -> Any:
    """Format message content."""
    if content and isinstance(content, list):
        # Remove unexpected block types
        formatted_content = []
        for block in content:
            if (
                isinstance(block, dict)
                and "type" in block
                and block["type"] == "tool_use"
            ):
                continue
            else:
                formatted_content.append(block)
    else:
        formatted_content = content

    return formatted_content


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_openai_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {"content": _format_message_content(message.content)}
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_openai_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            tool_call_supported_props = {"id", "type", "function"}
            message_dict["tool_calls"] = [
                {k: v for k, v in tool_call.items() if k in tool_call_supported_props}
                for tool_call in message_dict["tool_calls"]
            ]
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or ""
            message_dict["tool_calls"][0]["name"] = message_dict["tool_calls"][0][
                "function"
            ]["name"]
            message_dict["tool_calls"][0]["args"] = json.loads(
                message_dict["tool_calls"][0]["function"]["arguments"]
            )

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
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    id_ = "sample_id"
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("generated_text") or "")
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        try:
            tool_call_chunks = [
                {
                    "name": rtc["function"].get("name"),
                    "args": rtc["function"].get("arguments"),
                    "id": rtc.get("id"),
                    "index": rtc["index"],
                }
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
        )
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content, id=id_)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict["tool_call_id"], id=id_
        )
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    else:
        return default_class(content=content, id=id_)  # type: ignore


class _FunctionCall(TypedDict):
    name: str


class ChatWatsonx(BaseChatModel):
    """IBM watsonx.ai large language chat models.

    .. dropdown:: Setup
        :open:

        To use, you should have ``langchain_ibm`` python package installed,
        and the environment variable ``WATSONX_APIKEY`` set with your API key, or pass
        it as a named parameter to the constructor.

        .. code-block:: bash

            pip install -U langchain-ibm
            export WATSONX_APIKEY="your-api-key"


    Example:
        .. code-block:: python

            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
            parameters = {
                GenTextParamsMetaNames.DECODING_METHOD: "sample",
                GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
                GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
                GenTextParamsMetaNames.TEMPERATURE: 0.5,
                GenTextParamsMetaNames.TOP_K: 50,
                GenTextParamsMetaNames.TOP_P: 1,
            }

            from langchain_ibm import ChatWatsonx
            watsonx_llm = ChatWatsonx(
                model_id="meta-llama/llama-3-70b-instruct",
                url="https://us-south.ml.cloud.ibm.com",
                apikey="*****",
                project_id="*****",
                params=parameters,
            )
    """

    model_id: Optional[str] = None
    """Type of model to use."""

    deployment_id: Optional[str] = None
    """Type of deployed model to use."""

    project_id: Optional[str] = None
    """ID of the Watson Studio project."""

    space_id: Optional[str] = None
    """ID of the Watson Studio space."""

    url: SecretStr = Field(
        alias="url", default_factory=secret_from_env("WATSONX_URL", default=None)
    )
    """URL to the Watson Machine Learning or CPD instance."""

    apikey: Optional[SecretStr] = Field(
        alias="apikey", default_factory=secret_from_env("WATSONX_APIKEY", default=None)
    )
    """API key to the Watson Machine Learning or CPD instance."""

    token: Optional[SecretStr] = Field(
        alias="token", default_factory=secret_from_env("WATSONX_TOKEN", default=None)
    )
    """Token to the CPD instance."""

    password: Optional[SecretStr] = Field(
        alias="password",
        default_factory=secret_from_env("WATSONX_PASSWORD", default=None),
    )
    """Password to the CPD instance."""

    username: Optional[SecretStr] = Field(
        alias="username",
        default_factory=secret_from_env("WATSONX_USERNAME", default=None),
    )
    """Username to the CPD instance."""

    instance_id: Optional[SecretStr] = Field(
        alias="instance_id",
        default_factory=secret_from_env("WATSONX_INSTANCE_ID", default=None),
    )
    """Instance_id of the CPD instance."""

    version: Optional[SecretStr] = None
    """Version of the CPD instance."""

    params: Optional[dict] = None
    """Model parameters to use during request generation."""

    verify: Union[str, bool, None] = None
    """You can pass one of following as verify:
        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * True - default path to truststore will be taken
        * False - no verification will be made"""

    streaming: bool = False
    """ Whether to stream the results or not. """

    watsonx_model: ModelInference = Field(default=None, exclude=True)  #: :meta private:

    watsonx_client: Optional[APIClient] = Field(default=None, exclude=True)

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "watsonx-chat"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "ibm"
        if self.model_id:
            params["ls_model_name"] = self.model_id
        return params

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example:
            {
                "url": "WATSONX_URL",
                "apikey": "WATSONX_APIKEY",
                "token": "WATSONX_TOKEN",
                "password": "WATSONX_PASSWORD",
                "username": "WATSONX_USERNAME",
                "instance_id": "WATSONX_INSTANCE_ID",
            }
        """
        return {
            "url": "WATSONX_URL",
            "apikey": "WATSONX_APIKEY",
            "token": "WATSONX_TOKEN",
            "password": "WATSONX_PASSWORD",
            "username": "WATSONX_USERNAME",
            "instance_id": "WATSONX_INSTANCE_ID",
        }

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that credentials and python package exists in environment."""
        if isinstance(self.watsonx_client, APIClient):
            watsonx_model = ModelInference(
                model_id=self.model_id,
                params=self.params,
                api_client=self.watsonx_client,
                project_id=self.project_id,
                space_id=self.space_id,
                verify=self.verify,
            )
            self.watsonx_model = watsonx_model

        else:
            check_for_attribute(self.url, "url", "WATSONX_URL")

            if "cloud.ibm.com" in self.url.get_secret_value():
                check_for_attribute(self.apikey, "apikey", "WATSONX_APIKEY")
            else:
                if not self.token and not self.password and not self.apikey:
                    raise ValueError(
                        "Did not find 'token', 'password' or 'apikey',"
                        " please add an environment variable"
                        " `WATSONX_TOKEN`, 'WATSONX_PASSWORD' or 'WATSONX_APIKEY' "
                        "which contains it,"
                        " or pass 'token', 'password' or 'apikey'"
                        " as a named parameter."
                    )
                elif self.token:
                    check_for_attribute(self.token, "token", "WATSONX_TOKEN")
                elif self.password:
                    check_for_attribute(self.password, "password", "WATSONX_PASSWORD")
                    check_for_attribute(self.username, "username", "WATSONX_USERNAME")
                elif self.apikey:
                    check_for_attribute(self.apikey, "apikey", "WATSONX_APIKEY")
                    check_for_attribute(self.username, "username", "WATSONX_USERNAME")

                if not self.instance_id:
                    check_for_attribute(
                        self.instance_id, "instance_id", "WATSONX_INSTANCE_ID"
                    )

            credentials = Credentials(
                url=self.url.get_secret_value() if self.url else None,
                api_key=self.apikey.get_secret_value() if self.apikey else None,
                token=self.token.get_secret_value() if self.token else None,
                password=self.password.get_secret_value() if self.password else None,
                username=self.username.get_secret_value() if self.username else None,
                instance_id=self.instance_id.get_secret_value()
                if self.instance_id
                else None,
                version=self.version.get_secret_value() if self.version else None,
                verify=self.verify,
            )

            watsonx_chat = ModelInference(
                model_id=self.model_id,
                deployment_id=self.deployment_id,
                credentials=credentials,
                params=self.params,
                project_id=self.project_id,
                space_id=self.space_id,
            )
            self.watsonx_model = watsonx_chat

        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop, **kwargs)
        if message_dicts[-1].get("role") == "tool":
            chat_prompt = (
                "User: Please summarize given sentences into "
                "JSON containing Final Answer: '"
            )
            for message in message_dicts:
                if message["content"]:
                    chat_prompt += message["content"] + "\n"
            chat_prompt += "'"
        else:
            chat_prompt = self._create_chat_prompt(message_dicts)

        tools = kwargs.get("tools")

        if tools:
            chat_prompt = f"""
You are Mixtral Chat function calling, an AI language model developed by Mistral AI. 
You are a cautious assistant. You carefully follow instructions. You are helpful and 
harmless and you follow ethical guidelines and promote positive behavior. Here are a 
few of the tools available to you:
[AVAILABLE_TOOLS]
{json.dumps(tools[0], indent=2)}
[/AVAILABLE_TOOLS]
To use these tools you must always respond in JSON format containing `"type"` and 
`"function"` key-value pairs. Also `"function"` key-value pair always containing 
`"name"` and `"arguments"` key-value pairs. For example, to answer the question, 
"What is a length of word think?" you must use the get_word_length tool like so:

```json
{{
    "type": "function",
    "function": {{
        "name": "get_word_length",
        "arguments": {{
            "word": "think"
        }}
    }}
}}
```
</endoftext>

Remember, even when answering to the user, you must still use this JSON format! 
If you'd like to ask how the user is doing you must write:

```json
{{
    "type": "function",
    "function": {{
        "name": "Final Answer",
        "arguments": {{
            "output": "How are you today?"
        }}
    }}
}}
```
</endoftext>

Remember to end your response with '</endoftext>'

{chat_prompt}
(reminder to respond in a JSON blob no matter what and use tools only if necessary)"""

            params = params | {"stop_sequences": ["</endoftext>"]}

        if "tools" in kwargs:
            del kwargs["tools"]
        if "tool_choice" in kwargs:
            del kwargs["tool_choice"]

        response = self.watsonx_model.generate(
            prompt=chat_prompt, **(kwargs | {"params": params})
        )
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop, **kwargs)
        if message_dicts[-1].get("role") == "tool":
            chat_prompt = (
                "User: Please summarize given sentences into JSON "
                "containing Final Answer: '"
            )
            for message in message_dicts:
                if message["content"]:
                    chat_prompt += message["content"] + "\n"
            chat_prompt += "'"
        else:
            chat_prompt = self._create_chat_prompt(message_dicts)

        tools = kwargs.get("tools")

        if tools:
            chat_prompt = f"""
You are Mixtral Chat function calling, an AI language model developed by Mistral AI. 
You are a cautious assistant. You carefully follow instructions. You are helpful and 
harmless and you follow ethical guidelines and promote positive behavior. Here are a 
few of the tools available to you:
[AVAILABLE_TOOLS]
{json.dumps(tools[0], indent=2)}
[/AVAILABLE_TOOLS]
To use these tools you must always respond in JSON format containing `"type"` and 
`"function"` key-value pairs. Also `"function"` key-value pair always containing 
`"name"` and `"arguments"` key-value pairs. For example, to answer the question, 
"What is a length of word think?" you must use the get_word_length tool like so:

```json
{{
    "type": "function",
    "function": {{
        "name": "get_word_length",
        "arguments": {{
            "word": "think"
        }}
    }}
}}
```
</endoftext>

Remember, even when answering to the user, you must still use this JSON format! 
If you'd like to ask how the user is doing you must write:

```json
{{
    "type": "function",
    "function": {{
        "name": "Final Answer",
        "arguments": {{
            "output": "How are you today?"
        }}
    }}
}}
```
</endoftext>

Remember to end your response with '</endoftext>'

{chat_prompt[:-5]}
(reminder to respond in a JSON blob no matter what and use tools only if necessary)"""

            params = params | {"stop_sequences": ["</endoftext>"]}

        if "tools" in kwargs:
            del kwargs["tools"]
        if "tool_choice" in kwargs:
            del kwargs["tool_choice"]

        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk

        for chunk in self.watsonx_model.generate_text_stream(
            prompt=chat_prompt, raw_response=True, **(kwargs | {"params": params})
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["results"]) == 0:
                continue
            choice = chunk["results"][0]

            message_chunk = _convert_delta_to_message_chunk(choice, default_chunk_class)
            generation_info = {}
            if (finish_reason := choice.get("stop_reason")) != "not_finished":
                generation_info["finish_reason"] = finish_reason
            chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    def _create_chat_prompt(self, messages: List[Dict[str, Any]]) -> str:
        prompt = ""

        if self.model_id in ["ibm/granite-13b-chat-v1", "ibm/granite-13b-chat-v2"]:
            for message in messages:
                if message["role"] == "system":
                    prompt += "<|system|>\n" + message["content"] + "\n\n"
                elif message["role"] == "assistant":
                    prompt += "<|assistant|>\n" + message["content"] + "\n\n"
                elif message["role"] == "function":
                    prompt += "<|function|>\n" + message["content"] + "\n\n"
                elif message["role"] == "tool":
                    prompt += "<|tool|>\n" + message["content"] + "\n\n"
                else:
                    prompt += "<|user|>:\n" + message["content"] + "\n\n"

            prompt += "<|assistant|>\n"

        elif self.model_id in [
            "meta-llama/llama-2-13b-chat",
            "meta-llama/llama-2-70b-chat",
        ]:
            for message in messages:
                if message["role"] == "system":
                    prompt += "[INST] <<SYS>>\n" + message["content"] + "<</SYS>>\n\n"
                elif message["role"] == "assistant":
                    prompt += message["content"] + "\n[INST]\n\n"
                else:
                    prompt += message["content"] + "\n[/INST]\n"

        else:
            prompt = ChatPromptValue(
                messages=convert_to_messages(messages) + [AIMessage(content="")]
            ).to_string()

        return prompt

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]], **kwargs: Any
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = {**self.params} if self.params else {}
        params = params | {**kwargs.get("params", {})}
        if stop is not None:
            if params and "stop_sequences" in params:
                raise ValueError(
                    "`stop_sequences` found in both the input and default params."
                )
            params = (params or {}) | {"stop_sequences": stop}
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Union[dict]) -> ChatResult:
        generations = []
        sum_of_total_generated_tokens = 0
        sum_of_total_input_tokens = 0
        call_id = ""
        date_string = response.get("created_at")
        if date_string:
            date_object = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ")
            call_id = str(date_object.timestamp())

        if response.get("error"):
            raise ValueError(response.get("error"))

        for res in response["results"]:
            message = _convert_dict_to_message(res, call_id)
            generation_info = dict(finish_reason=res.get("stop_reason"))
            if "generated_token_count" in res:
                sum_of_total_generated_tokens += res["generated_token_count"]
            if "input_token_count" in res:
                sum_of_total_input_tokens += res["input_token_count"]
            total_token = sum_of_total_generated_tokens + sum_of_total_input_tokens
            if total_token and isinstance(message, AIMessage):
                message.usage_metadata = {
                    "input_tokens": sum_of_total_input_tokens,
                    "output_tokens": sum_of_total_generated_tokens,
                    "total_tokens": total_token,
                }
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = {
            "generated_token_count": sum_of_total_generated_tokens,
            "input_token_count": sum_of_total_input_tokens,
        }
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_id,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def bind_functions(
        self,
        functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        function_call: Optional[
            Union[_FunctionCall, str, Literal["auto", "none"]]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind functions (and other objects) to this chat model.

        Assumes model is compatible with IBM watsonx.ai function-calling API.

        Args:
            functions: A list of function definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, or callable. Pydantic
                models and callables will be automatically converted to
                their schema dictionary representation.
            function_call: Which function to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any).
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_functions = [convert_to_openai_function(fn) for fn in functions]
        if function_call is not None:
            function_call = (
                {"name": function_call}
                if isinstance(function_call, str)
                and function_call not in ("auto", "none")
                else function_call
            )
            if isinstance(function_call, dict) and len(formatted_functions) != 1:
                raise ValueError(
                    "When specifying `function_call`, you must provide exactly one "
                    "function."
                )
            if (
                isinstance(function_call, dict)
                and formatted_functions[0]["name"] != function_call["name"]
            ):
                raise ValueError(
                    f"Function call {function_call} was specified, but the only "
                    f"provided function was {formatted_functions[0]['name']}."
                )
            kwargs = {**kwargs, "function_call": function_call}
        return super().bind(
            functions=formatted_functions,
            **kwargs,
        )

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        bind_tools_supported_models = ["mistralai/mixtral-8x7b-instruct-v01"]
        if self.model_id not in bind_tools_supported_models:
            raise Warning(
                f"bind_tools() method for ChatWatsonx support only "
                f"following models: {bind_tools_supported_models}"
            )

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the IBM watsonx.ai function-calling spec.
            method: The method for steering model generation, either "function_calling"
                or "json_mode". If "function_calling" then the schema will be converted
                to an IBM watsonx.ai function and the returned model will make use of the
                function-calling API. If "json_mode" then IBM watsonx.ai's JSON mode will be
                used. Note that if using "json_mode" then you must include instructions
                for formatting the output into the desired schema into the model call.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_DictOrPydantic]
                    parsing_error: Optional[BaseException]

                If include_raw is False then just _DictOrPydantic is returned,
                where _DictOrPydantic depends on the schema:

                If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                    class.

                If schema is a dict then _DictOrPydantic is a dict.

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_ibm import ChatWatsonx
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatWatsonx(...)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=True):
            .. code-block:: python

                from langchain_ibm import ChatWatsonx
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatWatsonx(...)
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: Function-calling, dict schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_ibm import ChatWatsonx
                from pydantic import BaseModel
                from langchain_core.utils.function_calling import convert_to_openai_tool

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                dict_schema = convert_to_openai_tool(AnswerWithJustification)
                llm = ChatWatsonx(...)
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        Example: JSON mode, Pydantic schema (method="json_mode", include_raw=True):
            .. code-block::

                from langchain_ibm import ChatWatsonx
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatWatsonx(...)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_mode",
                    include_raw=True
                )

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'),
                #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
                #     'parsing_error': None
                # }

        Example: JSON mode, no schema (schema=None, method="json_mode", include_raw=True):
            .. code-block::

                from langchain_ibm import ChatWatsonx

                structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'),
                #     'parsed': {
                #         'answer': 'They are both the same weight.',
                #         'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'
                #     },
                #     'parsing_error': None
                # }
        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
            llm = self.bind_tools([schema], tool_choice=True)
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                key_name = convert_to_openai_tool(schema)["function"]["name"]
                output_parser = JsonOutputKeyToolsParser(
                    key_name=key_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(response_format={"type": "json_object"})
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_format'. Received: '{method}'"
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)
