"""IBM watsonx.ai large language chat models wrapper."""

import hashlib
import json
import logging
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
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
from ibm_watsonx_ai.foundation_models.schema import (  # type: ignore
    BaseSchema,
    TextChatParameters,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
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
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils._merge import merge_dicts
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_ibm.utils import (
    check_duplicate_chat_params,
    check_for_attribute,
    extract_chat_params,
)

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
    name = _dict.get("name")
    id_ = call_id
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), id=id_, name=name)
    elif role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: Dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    elif role == "system":
        return SystemMessage(content=_dict.get("content", ""), name=name, id=id_)
    elif role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=cast(str, _dict.get("name")), id=id_
        )
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=cast(str, _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )
    else:
        return ChatMessage(content=_dict.get("content", ""), role=role, id=id_)  # type: ignore[arg-type]


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


def _base62_encode(num: int) -> str:
    """Encodes a number in base62 and ensures result is of a specified length."""
    base62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if num == 0:
        return base62[0]
    arr = []
    base = len(base62)
    while num:
        num, rem = divmod(num, base)
        arr.append(base62[rem])
    arr.reverse()
    return "".join(arr)


def _convert_tool_call_id_to_mistral_compatible(tool_call_id: str) -> str:
    """Convert a tool call ID to a Mistral-compatible format"""
    hash_bytes = hashlib.sha256(tool_call_id.encode()).digest()
    hash_int = int.from_bytes(hash_bytes, byteorder="big")
    base62_str = _base62_encode(hash_int)
    if len(base62_str) >= 9:
        return base62_str[:9]
    else:
        return base62_str.rjust(9, "0")


def _convert_message_to_dict(message: BaseMessage, model_id: str | None) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.
        model_id: Type of model to use.

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
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None

        # Workaround for "mistralai/mistral-large" model when id < 9
        if model_id and model_id.startswith("mistralai"):
            tool_calls = message_dict.get("tool_calls", [])
            if (
                isinstance(tool_calls, list)
                and tool_calls
                and isinstance(tool_calls[0], dict)
            ):
                tool_call_id = tool_calls[0].get("id", "")
                if len(tool_call_id) < 9:
                    tool_call_id = _convert_tool_call_id_to_mistral_compatible(
                        tool_call_id
                    )

                message_dict["tool_calls"][0]["id"] = tool_call_id
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id

        # Workaround for "mistralai/mistral-large" model when tool_call_id < 9
        if model_id and model_id.startswith("mistralai"):
            tool_call_id = message_dict.get("tool_call_id", "")
            if len(tool_call_id) < 9:
                tool_call_id = _convert_tool_call_id_to_mistral_compatible(tool_call_id)

            message_dict["tool_call_id"] = tool_call_id

        supported_props = {"content", "role", "tool_call_id"}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any],
    default_class: Type[BaseMessageChunk],
    call_id: str,
    is_first_tool_chunk: bool,
) -> BaseMessageChunk:
    id_ = call_id
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
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


def _convert_chunk_to_generation_chunk(
    chunk: dict,
    default_chunk_class: Type,
    is_first_tool_chunk: bool,
    _prompt_tokens_included: bool,
) -> Optional[ChatGenerationChunk]:
    token_usage = chunk.get("usage")
    choices = chunk.get("choices", [])

    usage_metadata: Optional[UsageMetadata] = (
        _create_usage_metadata(token_usage, _prompt_tokens_included)
        if token_usage
        else None
    )

    if len(choices) == 0:
        # logprobs is implicitly None
        generation_chunk = ChatGenerationChunk(
            message=default_chunk_class(content="", usage_metadata=usage_metadata)
        )
        return generation_chunk

    choice = choices[0]
    if choice["delta"] is None:
        return None

    message_chunk = _convert_delta_to_message_chunk(
        choice["delta"], default_chunk_class, chunk["id"], is_first_tool_chunk
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

    generation_chunk = ChatGenerationChunk(
        message=message_chunk, generation_info=generation_info or None
    )
    return generation_chunk


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

            from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

            parameters = TextChatParameters(
                max_tokens=100,
                temperature=0.5,
                top_p=1,
                )

            from langchain_ibm import ChatWatsonx
            watsonx_llm = ChatWatsonx(
                model_id="meta-llama/llama-3-3-70b-instruct",
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
        alias="url",
        default_factory=secret_from_env("WATSONX_URL", default=None),  # type: ignore[assignment]
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

    params: Optional[Union[dict, TextChatParameters]] = None
    """Model parameters to use during request generation.

    Note:
        `ValueError` is raised if the same Chat generation parameter is provided 
        within the params attribute and as keyword argument."""

    frequency_penalty: Optional[float] = None
    """Positive values penalize new tokens based on their existing frequency in the 
    text so far, decreasing the model's likelihood to repeat the same line verbatim."""

    logprobs: Optional[bool] = None
    """Whether to return log probabilities of the output tokens or not. 
    If true, returns the log probabilities of each output token returned 
    in the content of message."""

    top_logprobs: Optional[int] = None
    """An integer specifying the number of most likely tokens to return at each 
    token position, each with an associated log probability. The option logprobs 
    must be set to true if this parameter is used."""

    max_tokens: Optional[int] = None
    """The maximum number of tokens that can be generated in the chat completion. 
    The total length of input tokens and generated tokens is limited by the 
    model's context length."""

    n: Optional[int] = None
    """How many chat completion choices to generate for each input message. 
    Note that you will be charged based on the number of generated tokens across 
    all of the choices. Keep n as 1 to minimize costs."""

    presence_penalty: Optional[float] = None
    """Positive values penalize new tokens based on whether they appear in the 
    text so far, increasing the model's likelihood to talk about new topics."""

    temperature: Optional[float] = None
    """What sampling temperature to use. Higher values like 0.8 will make the 
    output more random, while lower values like 0.2 will make it more focused 
    and deterministic.
    
    We generally recommend altering this or top_p but not both."""

    response_format: Optional[dict] = None
    """The chat response format parameters."""

    top_p: Optional[float] = None
    """An alternative to sampling with temperature, called nucleus sampling, 
    where the model considers the results of the tokens with top_p probability 
    mass. So 0.1 means only the tokens comprising the top 10% probability mass 
    are considered.

    We generally recommend altering this or temperature but not both."""

    time_limit: Optional[int] = None
    """Time limit in milliseconds - if not completed within this time, 
    generation will stop."""

    logit_bias: Optional[dict] = None
    """Increasing or decreasing probability of tokens being selected 
    during generation."""

    seed: Optional[int] = None
    """Random number generator seed to use in sampling mode 
    for experimental repeatability."""

    stop: Optional[list[str]] = None
    """Stop sequences are one or more strings which will cause the text generation 
    to stop if/when they are produced as part of the output."""

    verify: Union[str, bool, None] = None
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
            }
        )

        if isinstance(self.watsonx_client, APIClient):
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
            check_for_attribute(self.url, "url", "WATSONX_URL")

            if "cloud.ibm.com" in self.url.get_secret_value():
                if not self.token and not self.apikey:
                    raise ValueError(
                        "Did not find 'apikey' or 'token',"
                        " please add an environment variable"
                        " `WATSONX_APIKEY` or 'WATSONX_TOKEN' "
                        "which contains it,"
                        " or pass 'apikey' or 'token'"
                        " as a named parameter."
                    )
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
                verify=self.verify,
                validate=self.validate_model,
            )
            self.watsonx_model = watsonx_chat

        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop, **kwargs)
        updated_params = self._merge_params(params, kwargs)

        response = self.watsonx_model.chat(
            messages=message_dicts, **(kwargs | {"params": updated_params})
        )
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop, **kwargs)
        updated_params = self._merge_params(params, kwargs)

        response = await self.watsonx_model.achat(
            messages=message_dicts, **(kwargs | {"params": updated_params})
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
        updated_params = self._merge_params(params, kwargs)

        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk

        is_first_tool_chunk = True
        _prompt_tokens_included = False

        for chunk in self.watsonx_model.chat_stream(
            messages=message_dicts, **(kwargs | {"params": updated_params})
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            generation_chunk = _convert_chunk_to_generation_chunk(
                chunk, default_chunk_class, is_first_tool_chunk, _prompt_tokens_included
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
                    generation_chunk.text, chunk=generation_chunk, logprobs=logprobs
                )
            if hasattr(generation_chunk.message, "tool_calls") and isinstance(
                generation_chunk.message.tool_calls, list
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
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop, **kwargs)
        updated_params = self._merge_params(params, kwargs)

        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk

        is_first_tool_chunk = True
        _prompt_tokens_included = False

        response = await self.watsonx_model.achat_stream(
            messages=message_dicts, **(kwargs | {"params": updated_params})
        )
        async for chunk in response:
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            generation_chunk = _convert_chunk_to_generation_chunk(
                chunk,
                default_chunk_class,
                is_first_tool_chunk,
                _prompt_tokens_included,
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
                generation_chunk.message.tool_calls, list
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
    def _merge_params(params: dict, kwargs: dict) -> dict:
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
        self, messages: List[BaseMessage], stop: Optional[List[str]], **kwargs: Any
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = extract_chat_params(kwargs, self.params)

        if stop is not None:
            if params and "stop_sequences" in params:
                raise ValueError(
                    "`stop_sequences` found in both the input and default params."
                )
            params = (params or {}) | {"stop_sequences": stop}
        message_dicts = [_convert_message_to_dict(m, self.model_id) for m in messages]
        return message_dicts, params or {}

    def _create_chat_result(
        self, response: dict, generation_info: Optional[Dict] = None
    ) -> ChatResult:
        generations = []

        if response.get("error"):
            raise ValueError(response.get("error"))

        token_usage = response.get("usage", {})

        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"], response["id"])

            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(token_usage, False)
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
            "n",
            "presence_penalty",
            "response_format",
            "temperature",
            "top_p",
            "time_limit",
            "logit_bias",
            "seed",
            "stop",
        ]

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
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Options are:
                    - str of the form ``"<<tool_name>>"``: calls <<tool_name>> tool.
                    - ``"auto"``: automatically selects a tool (including no tool).
                    - ``"none"``: does not call a tool.
                    - ``"any"`` or ``"required"`` or ``True``: force at least one tool to be called.
                    - dict of the form ``{"type": "function", "function": {"name": <<tool_name>>}}``: calls <<tool_name>> tool.
                    - ``False`` or ``None``: no effect, default OpenAI behavior.

            kwargs: Any additional parameters are passed directly to
                ``self.bind(**kwargs)``.
        """  # noqa: E501
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
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
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )

            if isinstance(tool_choice, str):
                kwargs["tool_choice_option"] = tool_choice
            else:
                kwargs["tool_choice"] = tool_choice
        else:
            kwargs["tool_choice_option"] = "auto"

        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type]] = None,
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
        is_pydantic_schema = isinstance(schema, type) and is_basemodel_subclass(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
            # specifying a tool.
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            tool_choice = {"type": "function", "function": {"name": tool_name}}
            llm = self.bind_tools([schema], tool_choice=tool_choice)
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


def _lc_tool_call_to_watsonx_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_watsonx_tool_call(
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


def _create_usage_metadata(
    oai_token_usage: dict,
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
