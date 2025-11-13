"""Base classes for IBM watsonx.ai large language models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from ibm_watsonx_ai import APIClient  # type: ignore[import-untyped]
from ibm_watsonx_ai.foundation_models import (  # type: ignore[import-untyped]
    Model,
    ModelInference,
)
from ibm_watsonx_ai.gateway import Gateway  # type: ignore[import-untyped]
from ibm_watsonx_ai.metanames import (  # type: ignore[import-untyped]
    GenTextParamsMetaNames,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.utils.utils import secret_from_env
from pydantic import AliasChoices, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_ibm.utils import (
    async_gateway_error_handler,
    extract_params,
    gateway_error_handler,
    normalize_api_key,
    resolve_watsonx_credentials,
    secret_from_env_multi,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping

    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )

logger = logging.getLogger(__name__)
textgen_valid_params = [
    value for key, value in GenTextParamsMetaNames.__dict__.items() if key.isupper()
]


class WatsonxLLM(BaseLLM):
    """`IBM watsonx.ai` large language models class.

    ???+ info "Setup"

        To use the large language models, you need to have the `langchain_ibm` python
        package installed, and the environment variable `WATSONX_API_KEY` set with your
        API key or pass it as a named parameter `api_key` to the constructor.

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

        ```python
        from langchain_ibm import WatsonxLLM
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

        parameters = {
            GenTextParamsMetaNames.DECODING_METHOD: "sample",
            GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
            GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
            GenTextParamsMetaNames.TEMPERATURE: 0.5,
            GenTextParamsMetaNames.TOP_K: 50,
            GenTextParamsMetaNames.TOP_P: 1,
        }

        model = WatsonxLLM(
            model_id="google/flan-t5-xl",
            url="https://us-south.ml.cloud.ibm.com",
            project_id="*****",
            params=parameters,
            # api_key="*****"
        )
        ```

    ??? info "Invoke"

        ```python
        input_text = "The meaning of life is "
        response = model.invoke(input_text)
        print(response)
        ```

        ```txt
        "42, but what was the question?
        The answer to the ultimate question of life, the universe, and everything is 42.
        But what was the question? This is a reference to Douglas Adams' science fiction
        series "The Hitchhiker's Guide to the Galaxy."
        ```

    ??? info "Stream"

        ```python
        for chunk in model.stream(input_text):
            print(chunk, end="")
        ```

        ```txt
        "42, but what was the question?
        The answer to the ultimate question of life, the universe, and everything is 42.
        But what was the question? This is a reference to Douglas Adams' science fiction
        series "The Hitchhiker's Guide to the Galaxy."
        ```

    ??? info "Async"
        ```python
        response = await model.ainvoke(input_text)

        # stream:
        # async for chunk in model.astream(input_text):
        #     print(chunk, end="")

        # batch:
        # await model.abatch([input_text])
        ```

        ```txt
        "42, but what was the question?
        The answer to the ultimate question of life, the universe, and everything is 42.
        But what was the question? This is a reference to Douglas Adams' science fiction
        series "The Hitchhiker's Guide to the Galaxy."
        ```
    """

    model_id: str | None = None
    """Type of model to use."""

    model: str | None = None
    """
    Name or alias of the foundation model to use.
    When using IBM's watsonx.ai Model Gateway (public preview), you can specify any
    supported third-party model-OpenAI, Anthropic, NVIDIA, Cerebras, or IBM's own
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

    params: dict[str, Any] | None = None
    """Model parameters to use during request generation."""

    verify: str | bool | None = None
    """You can pass one of following as verify:
        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * True - default path to truststore will be taken
        * False - no verification will be made"""

    streaming: bool = False
    """ Whether to stream the results or not. """

    watsonx_model: ModelInference = Field(default=None, exclude=True)  #: :meta private:

    watsonx_model_gateway: Gateway = Field(
        default=None,
        exclude=True,
    )  #: :meta private:

    watsonx_client: APIClient | None = Field(default=None)

    model_config = ConfigDict(
        extra="forbid",
    )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Is lc serializable."""
        return False

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
        if self.watsonx_model_gateway is not None:
            error_msg = (
                "Passing the 'watsonx_model_gateway' parameter to the WatsonxLLM "
                "constructor is not supported yet.",
            )
            raise NotImplementedError(error_msg)

        if isinstance(self.watsonx_model, (ModelInference, Model)):
            self.model_id = self.watsonx_model.model_id
            self.deployment_id = getattr(self.watsonx_model, "deployment_id", "")
            self.project_id = self.watsonx_model._client.default_project_id  # noqa: SLF001
            self.space_id = self.watsonx_model._client.default_space_id  # noqa: SLF001
            self.params = self.watsonx_model.params

        elif isinstance(self.watsonx_client, APIClient):
            if sum(map(bool, (self.model, self.model_id, self.deployment_id))) != 1:
                error_msg = (
                    "The parameters 'model', 'model_id' and 'deployment_id' are "
                    "mutually exclusive. Please specify exactly one of these "
                    "parameters when initializing WatsonxLLM.",
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
                )
                self.watsonx_model = watsonx_model

        else:
            if sum(map(bool, (self.model, self.model_id, self.deployment_id))) != 1:
                error_msg = (
                    "The parameters 'model', 'model_id' and 'deployment_id' are "
                    "mutually exclusive. Please specify exactly one of these "
                    "parameters when initializing WatsonxLLM.",
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
                )
                self.watsonx_model = watsonx_model

        return self

    @gateway_error_handler
    def _call_model_gateway(
        self, *, model: str, prompt: str | list[str] | list[int], **params: Any
    ) -> Any:
        return self.watsonx_model_gateway.completions.create(
            model=model,
            prompt=prompt,
            **params,
        )

    @async_gateway_error_handler
    async def _acall_model_gateway(
        self,
        *,
        model: str,
        prompt: str | list[str] | list[int],
        **params: Any,
    ) -> Any:
        return await self.watsonx_model_gateway.completions.acreate(
            model=model,
            prompt=prompt,
            **params,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "deployment_id": self.deployment_id,
            "params": self.params,
            "project_id": self.project_id,
            "space_id": self.space_id,
        }

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "IBM watsonx.ai"

    @staticmethod
    def _extract_token_usage(
        response: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if response is None:
            return {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        completion_tokens = 0
        prompt_tokens = 0

        def get_count_value(key: str, result: dict[str, Any]) -> int:
            return result.get(key, 0) or 0

        for res in response:
            results = res.get("results")
            if results:
                prompt_tokens += get_count_value("input_token_count", results[0])
                completion_tokens += get_count_value(
                    "generated_token_count",
                    results[0],
                )
        total_tokens = completion_tokens + prompt_tokens
        return {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _validate_chat_params(
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate and fix the chat parameters."""
        for param in params:
            if param.lower() not in textgen_valid_params:
                error_msg = (
                    f"Parameter {param} is not valid. "
                    f"Valid parameters are: {textgen_valid_params}"
                )
                raise ValueError(error_msg)
        return params

    @staticmethod
    def _override_chat_params(
        params: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Override class parameters with those provided in the invoke method.

        Merges the 'params' dictionary with any 'params' found in kwargs,
        then updates 'params' with matching keys from kwargs and removes
        those keys from kwargs.
        """
        for key in list(kwargs.keys()):
            if key.lower() in textgen_valid_params:
                params[key] = kwargs.pop(key)
        return params, kwargs

    def _get_chat_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        params = extract_params(kwargs, self.params)

        params, kwargs = self._override_chat_params(params or {}, **kwargs)
        if stop is not None:
            if params and "stop_sequences" in params:
                error_msg = (
                    "`stop_sequences` found in both the input and default params."
                )
                raise ValueError(error_msg)
            params = (params or {}) | {"stop_sequences": stop}
        return params, kwargs

    def _create_llm_result(self, response: list[dict[str, Any]]) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        generations = [
            [
                Generation(
                    text=result.get("generated_text", ""),
                    generation_info={"finish_reason": result.get("stop_reason")}
                    | (
                        {"moderations": moderations}
                        if (moderations := result.get("moderations"))
                        else {}
                    ),
                ),
            ]
            for res in response
            if (results := res.get("results"))
            for result in results
        ]
        llm_output = {
            "token_usage": self._extract_token_usage(response),
            "model_id": self.model_id,
            "deployment_id": self.deployment_id,
        }
        return LLMResult(generations=generations, llm_output=llm_output)

    def _create_llm_gateway_result(self, response: dict[str, Any]) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        choices = response["choices"]

        generations = [
            [
                Generation(
                    text=choice["text"],
                    generation_info={
                        "finish_reason": choice.get("finish_reason"),
                        "logprobs": choice.get("logprobs"),
                    },
                ),
            ]
            for choice in choices
        ]

        llm_output = {
            "token_usage": response["usage"]["total_tokens"],
            "model_id": self.model_id,
            "deployment_id": self.deployment_id,
        }
        return LLMResult(generations=generations, llm_output=llm_output)

    def _stream_response_to_generation_chunk(
        self,
        stream_response: dict[str, Any],
    ) -> GenerationChunk:
        """Convert a stream response to a generation chunk."""
        result = stream_response.get("results", [{}])[0]
        if not result:
            return GenerationChunk(text="")

        finish_reason = result.get("stop_reason")
        finish_reason = None if finish_reason == "not_finished" else finish_reason

        generation_info = {
            "finish_reason": finish_reason,
            "llm_output": {
                "model_id": self.model_id,
                "deployment_id": self.deployment_id,
            },
        }

        if moderations := result.get("moderations"):
            generation_info["moderations"] = moderations

        return GenerationChunk(
            text=result.get("generated_text", ""),
            generation_info=generation_info,
        )

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Call the IBM watsonx.ai inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating the response.
            run_manager: Optional callback manager.
            kwargs: Additional keyword args

        Returns:
            The string generated by the model.

        Example:
            ```python
            response = model.invoke("What is a molecule")
            ```
        """
        result = self._generate(
            prompts=[prompt],
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        return result.generations[0][0].text

    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Async version of the _call method."""
        result = await self._agenerate(
            prompts=[prompt],
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        return result.generations[0][0].text

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call the IBM watsonx.ai inference endpoint that then generates the response.

        Args:
            prompts: List of strings (prompts) to pass into the model
            stop: Optional list of stop words to use when generating the response
            run_manager: Optional callback manager
            stream: Stream response
            kwargs: Additional keyword args

        Returns:
            The full LLMResult output.

        Example:
            ```python
            response = model.generate(["What is a molecule"])
            ```
        """
        params, kwargs = self._get_chat_params(stop=stop, **kwargs)
        params = self._validate_chat_params(params)
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            if len(prompts) > 1:
                error_msg = (
                    f"WatsonxLLM currently only supports single prompt, got {prompts}"
                )
                raise ValueError(error_msg)
            generation = GenerationChunk(text="")
            stream_iter = self._stream(
                prompts[0],
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            for chunk in stream_iter:
                if generation is None:
                    generation = chunk  # type: ignore[unreachable]
                else:
                    generation += chunk
            if generation is None:
                error_msg = "No generation chunks were received from the stream."  # type: ignore[unreachable]
                raise RuntimeError(error_msg)
            if isinstance(generation.generation_info, dict):
                llm_output = generation.generation_info.pop("llm_output")
                return LLMResult(generations=[[generation]], llm_output=llm_output)
            return LLMResult(generations=[[generation]])
        if self.watsonx_model_gateway is not None:
            call_kwargs = {**kwargs, **params}
            response = self._call_model_gateway(
                model=self.model,
                prompt=prompts,
                **call_kwargs,
            )
            return self._create_llm_gateway_result(response)
        response = self.watsonx_model.generate(prompt=prompts, params=params, **kwargs)
        return self._create_llm_result(response)

    async def _agenerate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async run the LLM on the given prompt and input."""
        params, kwargs = self._get_chat_params(stop=stop, **kwargs)
        params = self._validate_chat_params(params)
        if stream:
            return await super()._agenerate(
                prompts=prompts,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        if self.watsonx_model_gateway is not None:
            call_kwargs = {**kwargs, **params}
            responses = await self._acall_model_gateway(
                model=self.model,
                prompt=prompts,
                **call_kwargs,
            )
            return self._create_llm_gateway_result(responses)
        responses = [
            await self.watsonx_model.agenerate(prompt=prompt, params=params, **kwargs)
            for prompt in prompts
        ]

        return self._create_llm_result(responses)

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call the IBM watsonx.ai inference endpoint that then streams the response.

        Args:
            prompt: The prompt to pass into the model
            stop: Optional list of stop words to use when generating the response
            run_manager: Optional callback manager
            kwargs: Additional keyword args

        Returns:
            The iterator which yields generation chunks

        Example:
            ```python
            response = model.stream("What is a molecule")
            for chunk in response:
                print(chunk, end="", flush=True)
            ```
        """
        params, kwargs = self._get_chat_params(stop=stop, **kwargs)
        params = self._validate_chat_params(params)
        if self.watsonx_model_gateway is not None:
            call_kwargs = {**kwargs, **params, "stream": True}
            chunk_iter = self._call_model_gateway(
                model=self.model,
                prompt=prompt,
                **call_kwargs,
            )
        else:
            chunk_iter = self.watsonx_model.generate_text_stream(
                prompt=prompt,
                params=params,
                **(kwargs | {"raw_response": True}),
            )
        for stream_resp in chunk_iter:
            if not isinstance(stream_resp, dict):
                stream_data = stream_resp.dict()
            else:
                stream_data = stream_resp
            chunk = self._stream_response_to_generation_chunk(stream_data)

            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        params, kwargs = self._get_chat_params(stop=stop, **kwargs)
        params = self._validate_chat_params(params)

        if self.watsonx_model_gateway is not None:
            call_kwargs = {**kwargs, **params, "stream": True}
            chunk_iter = await self._acall_model_gateway(
                model=self.model,
                prompt=prompt,
                **call_kwargs,
            )
        else:
            chunk_iter = await self.watsonx_model.agenerate_stream(
                prompt=prompt,
                params=params,
            )
        async for stream_resp in chunk_iter:
            if not isinstance(stream_resp, dict):
                stream_data = stream_resp.dict()
            else:
                stream_data = stream_resp
            chunk = self._stream_response_to_generation_chunk(stream_data)

            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def get_num_tokens(self, text: str) -> int:
        """Get num tokens."""
        if self.watsonx_model_gateway is not None:
            error_msg = (
                "Tokenize endpoint is not supported by IBM Model Gateway endpoint."
            )
            raise NotImplementedError(error_msg)
        response = self.watsonx_model.tokenize(text, return_tokens=False)
        return cast("int", response["result"]["token_count"])

    def get_token_ids(self, text: str) -> list[int]:
        """Get token ids."""
        error_msg = "API does not support returning token ids."
        raise NotImplementedError(error_msg)
