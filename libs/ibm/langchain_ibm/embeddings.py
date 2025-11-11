"""IBM watsonx.ai embeddings wrapper."""

import logging
from typing import Any, cast

from ibm_watsonx_ai import APIClient  # type: ignore[import-untyped]
from ibm_watsonx_ai.foundation_models.embeddings import (  # type: ignore[import-untyped]
    Embeddings,
)
from ibm_watsonx_ai.gateway import Gateway  # type: ignore[import-untyped]
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from langchain_core.utils.utils import secret_from_env
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

from langchain_ibm.utils import (
    async_gateway_error_handler,
    extract_params,
    gateway_error_handler,
    normalize_api_key,
    resolve_watsonx_credentials,
    secret_from_env_multi,
)

logger = logging.getLogger(__name__)


class WatsonxEmbeddings(BaseModel, LangChainEmbeddings):
    """`IBM watsonx.ai` embedding model integration.

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

        ```python
        from langchain_ibm import WatsonxEmbeddings

        embeddings = WatsonxEmbeddings(
            model_id="ibm/granite-embedding-278m-multilingual",
            url="https://us-south.ml.cloud.ibm.com",
            project_id="*****",
            # api_key="*****"
        )
        ```

    ??? info "Embed single text"

        ```python
        input_text = "The meaning of life is 42"
        vector = embeddings.embed_query("hello")
        print(vector[:3])
        ```

        ```python
        [-0.0020519258, 0.0147288125, -0.0090887165]
        ```

    ??? info "Embed multiple texts"

        ```python
        vectors = embeddings.embed_documents(["hello", "goodbye"])
        # Showing only the first 3 coordinates
        print(len(vectors))
        print(vectors[0][:3])
        ```

        ```python
        2
        [-0.0020519265, 0.01472881, -0.009088721]
        ```

    ??? info "Async"

        ```python
        await embeddings.aembed_query(input_text)
        print(vector[:3])

        # multiple:
        # await embeddings.aembed_documents(input_texts)
        ```

        ```python
        [-0.0020519258, 0.0147288125, -0.0090887165]
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
        * False - no verification will be made
    """

    watsonx_embed: Embeddings = Field(default=None)  #: :meta private:

    watsonx_embed_gateway: Gateway = Field(
        default=None,
        exclude=True,
    )  #: :meta private:

    watsonx_client: APIClient | None = Field(default=None)  #: :meta private:

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

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
        if self.watsonx_embed_gateway is not None:
            error_msg = (
                "Passing the 'watsonx_embed_gateway' parameter to the "
                "WatsonxEmbeddings constructor is not supported yet.",
            )
            raise NotImplementedError(error_msg)

        if isinstance(self.watsonx_embed, Embeddings):
            self.model_id = self.watsonx_embed.model_id
            self.project_id = self.watsonx_embed._client.default_project_id  # noqa: SLF001
            self.space_id = self.watsonx_embed._client.default_space_id  # noqa: SLF001

            self.params = self.watsonx_embed.params

        elif isinstance(self.watsonx_client, APIClient):
            if sum(map(bool, (self.model, self.model_id))) != 1:
                error_msg = (
                    "The parameters 'model' and 'model_id' are mutually exclusive. "
                    "Please specify exactly one of these parameters when "
                    "initializing WatsonxEmbeddings.",
                )
                raise ValueError(error_msg)
            if self.model is not None:
                watsonx_embed_gateway = Gateway(
                    api_client=self.watsonx_client,
                    verify=self.verify,
                )
                self.watsonx_embed_gateway = watsonx_embed_gateway
            else:
                watsonx_embed = Embeddings(
                    model_id=self.model_id,
                    params=self.params,
                    api_client=self.watsonx_client,
                    project_id=self.project_id,
                    space_id=self.space_id,
                    verify=self.verify,
                )
                self.watsonx_embed = watsonx_embed

        else:
            if sum(map(bool, (self.model, self.model_id))) != 1:
                error_msg = (
                    "The parameters 'model' and 'model_id' are mutually exclusive. "
                    "Please specify exactly one of these parameters when "
                    "initializing WatsonxEmbeddings.",
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
                watsonx_embed_gateway = Gateway(
                    credentials=credentials,
                    verify=self.verify,
                )
                self.watsonx_embed_gateway = watsonx_embed_gateway

            else:
                watsonx_embed = Embeddings(
                    model_id=self.model_id,
                    params=self.params,
                    credentials=credentials,
                    project_id=self.project_id,
                    space_id=self.space_id,
                )

                self.watsonx_embed = watsonx_embed

        return self

    @gateway_error_handler
    def _call_model_gateway(
        self,
        *,
        model: str,
        texts: list[str],
        **params: Any,
    ) -> Any:
        return self.watsonx_embed_gateway.embeddings.create(
            model=model,
            input=texts,
            **params,
        )

    @async_gateway_error_handler
    async def _acall_model_gateway(
        self,
        *,
        model: str,
        texts: list[str],
        **params: Any,
    ) -> Any:
        return await self.watsonx_embed_gateway.embeddings.acreate(
            model=model,
            input=texts,
            **params,
        )

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Embed search docs."""
        params = extract_params(kwargs, self.params)
        if self.watsonx_embed_gateway is not None:
            call_kwargs = {**kwargs, **params}
            embed_response = self._call_model_gateway(
                model=self.model,
                texts=texts,
                **call_kwargs,
            )
            return [embedding["embedding"] for embedding in embed_response["data"]]
        return cast(
            "list[list[float]]",
            self.watsonx_embed.embed_documents(
                texts=texts,
                **(kwargs | {"params": params}),
            ),
        )

    async def aembed_documents(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """Asynchronous Embed search docs."""
        params = extract_params(kwargs, self.params)
        if self.watsonx_embed_gateway is not None:
            call_kwargs = {**kwargs, **params}
            embed_response = await self._acall_model_gateway(
                model=self.model,
                texts=texts,
                **call_kwargs,
            )
            return [embedding["embedding"] for embedding in embed_response["data"]]
        return cast(
            "list[list[float]]",
            await self.watsonx_embed.aembed_documents(
                texts=texts,
                **(kwargs | {"params": params}),
            ),
        )

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Embed query text."""
        return self.embed_documents([text], **kwargs)[0]

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Asynchronous Embed query text."""
        embeddings = await self.aembed_documents([text], **kwargs)
        return embeddings[0]
