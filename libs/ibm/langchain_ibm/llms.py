from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
from ibm_watsonx_ai.foundation_models import Model, ModelInference  # type: ignore
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames  # type: ignore
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.utils.utils import secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_ibm.utils import check_for_attribute, extract_params

logger = logging.getLogger(__name__)
textgen_valid_params = [
    value for key, value in GenTextParamsMetaNames.__dict__.items() if key.isupper()
]


class WatsonxLLM(BaseLLM):
    """
    IBM watsonx.ai large language models.

    To use the large language models, you need to have the ``langchain_ibm``
    python package installed, and the environment variable ``WATSONX_APIKEY``
    set with your API key or pass it as a named parameter to the constructor.


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

            from langchain_ibm import WatsonxLLM
            watsonx_llm = WatsonxLLM(
                model_id="google/flan-ul2",
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

    watsonx_client: Optional[APIClient] = Field(default=None)

    model_config = ConfigDict(
        extra="forbid",
    )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names for secret IDs.

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
        if isinstance(self.watsonx_model, (ModelInference, Model)):
            self.model_id = getattr(self.watsonx_model, "model_id")
            self.deployment_id = getattr(self.watsonx_model, "deployment_id", "")
            self.project_id = getattr(
                getattr(self.watsonx_model, "_client"),
                "default_project_id",
            )
            self.space_id = getattr(
                getattr(self.watsonx_model, "_client"), "default_space_id"
            )
            self.params = getattr(self.watsonx_model, "params")

        elif isinstance(self.watsonx_client, APIClient):
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
        response: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if response is None:
            return {"generated_token_count": 0, "input_token_count": 0}

        input_token_count = 0
        generated_token_count = 0

        def get_count_value(key: str, result: Dict[str, Any]) -> int:
            return result.get(key, 0) or 0

        for res in response:
            results = res.get("results")
            if results:
                input_token_count += get_count_value("input_token_count", results[0])
                generated_token_count += get_count_value(
                    "generated_token_count", results[0]
                )

        return {
            "generated_token_count": generated_token_count,
            "input_token_count": input_token_count,
        }

    @staticmethod
    def _validate_chat_params(
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate and fix the chat parameters."""
        for param in params.keys():
            if param.lower() not in textgen_valid_params:
                raise Exception(
                    f"Parameter {param} is not valid. "
                    f"Valid parameters are: {textgen_valid_params}"
                )
        return params

    @staticmethod
    def _override_chat_params(
        params: Dict[str, Any], **kwargs: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Override class parameters with those provided in the invoke method.
        Merges the 'params' dictionary with any 'params' found in kwargs,
        then updates 'params' with matching keys from kwargs and removes
        those keys from kwargs.
        """
        for key in list(kwargs.keys()):
            if key.lower() in textgen_valid_params:
                params[key] = kwargs.pop(key)
        return params, kwargs

    def _get_chat_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        params = extract_params(kwargs, self.params)

        params, kwargs = self._override_chat_params(params or {}, **kwargs)
        if stop is not None:
            if params and "stop_sequences" in params:
                raise ValueError(
                    "`stop_sequences` found in both the input and default params."
                )
            params = (params or {}) | {"stop_sequences": stop}
        return params, kwargs

    def _create_llm_result(self, response: List[dict]) -> LLMResult:
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
                )
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

    def _stream_response_to_generation_chunk(
        self,
        stream_response: Dict[str, Any],
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
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the IBM watsonx.ai inference endpoint.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating the response.
            run_manager: Optional callback manager.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python

                response = watsonx_llm.invoke("What is a molecule")
        """
        result = self._generate(
            prompts=[prompt], stop=stop, run_manager=run_manager, **kwargs
        )
        return result.generations[0][0].text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async version of the _call method."""

        result = await self._agenerate(
            prompts=[prompt], stop=stop, run_manager=run_manager, **kwargs
        )
        return result.generations[0][0].text

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call the IBM watsonx.ai inference endpoint that then generates the response.
        Args:
            prompts: List of strings (prompts) to pass into the model.
            stop: Optional list of stop words to use when generating the response.
            run_manager: Optional callback manager.
        Returns:
            The full LLMResult output.
        Example:
            .. code-block:: python

                response = watsonx_llm.generate(["What is a molecule"])
        """
        params, kwargs = self._get_chat_params(stop=stop, **kwargs)
        params = self._validate_chat_params(params)
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            if len(prompts) > 1:
                raise ValueError(
                    f"WatsonxLLM currently only supports single prompt, got {prompts}"
                )
            generation = GenerationChunk(text="")
            stream_iter = self._stream(
                prompts[0], stop=stop, run_manager=run_manager, **kwargs
            )
            for chunk in stream_iter:
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            if isinstance(generation.generation_info, dict):
                llm_output = generation.generation_info.pop("llm_output")
                return LLMResult(generations=[[generation]], llm_output=llm_output)
            return LLMResult(generations=[[generation]])
        else:
            response = self.watsonx_model.generate(
                prompt=prompts, params=params, **kwargs
            )
            return self._create_llm_result(response)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async run the LLM on the given prompt and input."""
        params, kwargs = self._get_chat_params(stop=stop, **kwargs)
        params = self._validate_chat_params(params)
        if stream:
            return await super()._agenerate(
                prompts=prompts, stop=stop, run_manager=run_manager, **kwargs
            )
        else:
            responses = [
                await self.watsonx_model.agenerate(
                    prompt=prompt, params=params, **kwargs
                )
                for prompt in prompts
            ]

            return self._create_llm_result(responses)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call the IBM watsonx.ai inference endpoint that then streams the response.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating the response.
            run_manager: Optional callback manager.
        Returns:
            The iterator which yields generation chunks.
        Example:
            .. code-block:: python

                response = watsonx_llm.stream("What is a molecule")
                for chunk in response:
                    print(chunk, end='', flush=True)
        """
        params, kwargs = self._get_chat_params(stop=stop, **kwargs)
        params = self._validate_chat_params(params)
        for stream_resp in self.watsonx_model.generate_text_stream(
            prompt=prompt, params=params, **(kwargs | {"raw_response": True})
        ):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.dict()
            chunk = self._stream_response_to_generation_chunk(stream_resp)

            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        params, kwargs = self._get_chat_params(stop=stop, **kwargs)
        params = self._validate_chat_params(params)
        async for stream_resp in await self.watsonx_model.agenerate_stream(
            prompt=prompt, params=params
        ):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.dict()
            chunk = self._stream_response_to_generation_chunk(stream_resp)

            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def get_num_tokens(self, text: str) -> int:
        response = self.watsonx_model.tokenize(text, return_tokens=False)
        return response["result"]["token_count"]

    def get_token_ids(self, text: str) -> List[int]:
        raise NotImplementedError("API does not support returning token ids.")
