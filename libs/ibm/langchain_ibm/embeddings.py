import logging
from typing import Any, Dict, List, Optional, Union

from ibm_watsonx_ai import APIClient  # type: ignore
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings  # type: ignore
from ibm_watsonx_ai.gateway import Gateway  # type: ignore
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from langchain_core.utils.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_ibm.utils import (
    async_gateway_error_handler,
    extract_params,
    gateway_error_handler,
    resolve_watsonx_credentials,
)

logger = logging.getLogger(__name__)


class WatsonxEmbeddings(BaseModel, LangChainEmbeddings):
    """IBM watsonx.ai embedding models."""

    model_id: Optional[str] = None
    """Type of model to use."""

    model: Optional[str] = None
    """
    Name or alias of the foundation model to use.  
    When using IBM’s watsonx.ai Model Gateway (public preview), you can specify any 
    supported third-party model—OpenAI, Anthropic, NVIDIA, Cerebras, or IBM’s own 
    Granite series—via a single, OpenAI-compatible interface. Models must be explicitly 
    provisioned (opt-in) through the Gateway to ensure secure, vendor-agnostic access 
    and easy switch-over without reconfiguration.
    
    For more details on configuration and usage, see IBM watsonx Model Gateway docs: https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-model-gateway.html?context=wx&audience=wdp
    """

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

    params: Optional[Dict] = None
    """Model parameters to use during request generation."""

    verify: Union[str, bool, None] = None
    """You can pass one of following as verify:
        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * True - default path to truststore will be taken
        * False - no verification will be made
    """

    watsonx_embed: Embeddings = Field(default=None)  #: :meta private:

    watsonx_embed_gateway: Gateway = Field(
        default=None, exclude=True
    )  #: :meta private:

    watsonx_client: Optional[APIClient] = Field(default=None)  #: :meta private:

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that credentials and python package exists in environment."""
        if self.watsonx_embed_gateway is not None:
            raise NotImplementedError(
                "Passing the 'watsonx_embed_gateway' parameter to the "
                "WatsonxEmbeddings constructor is not supported yet."
            )

        if isinstance(self.watsonx_embed, Embeddings):
            self.model_id = getattr(self.watsonx_embed, "model_id")
            self.project_id = getattr(
                getattr(self.watsonx_embed, "_client"),
                "default_project_id",
            )
            self.space_id = getattr(
                getattr(self.watsonx_embed, "_client"), "default_space_id"
            )

            self.params = getattr(self.watsonx_embed, "params")

        elif isinstance(self.watsonx_client, APIClient):
            if sum(map(bool, (self.model, self.model_id))) != 1:
                raise ValueError(
                    "The parameters 'model' and 'model_id' are mutually exclusive. "
                    "Please specify exactly one of these parameters when "
                    "initializing WatsonxEmbeddings."
                )
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
                raise ValueError(
                    "The parameters 'model' and 'model_id' are mutually exclusive. "
                    "Please specify exactly one of these parameters when "
                    "initializing WatsonxEmbeddings."
                )
            credentials = resolve_watsonx_credentials(
                url=self.url,
                apikey=self.apikey,
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
        self, *, model: str, texts: List[str], **params: Any
    ) -> Any:
        return self.watsonx_embed_gateway.embeddings.create(
            model=model, input=texts, **params
        )

    @async_gateway_error_handler
    async def _acall_model_gateway(
        self, *, model: str, texts: List[str], **params: Any
    ) -> Any:
        return await self.watsonx_embed_gateway.embeddings.acreate(
            model=model, input=texts, **params
        )

    def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Embed search docs."""
        params = extract_params(kwargs, self.params)
        if self.watsonx_embed_gateway is not None:
            call_kwargs = {**kwargs, **params}
            embed_response = self._call_model_gateway(
                model=self.model, texts=texts, **call_kwargs
            )
            return [embedding["embedding"] for embedding in embed_response["data"]]
        else:
            return self.watsonx_embed.embed_documents(
                texts=texts, **(kwargs | {"params": params})
            )

    async def aembed_documents(
        self, texts: List[str], **kwargs: Any
    ) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        params = extract_params(kwargs, self.params)
        if self.watsonx_embed_gateway is not None:
            call_kwargs = {**kwargs, **params}
            embed_response = await self._acall_model_gateway(
                model=self.model, texts=texts, **call_kwargs
            )
            return [embedding["embedding"] for embedding in embed_response["data"]]
        else:
            return await self.watsonx_embed.aembed_documents(
                texts=texts, **(kwargs | {"params": params})
            )

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text], **kwargs)[0]

    async def aembed_query(self, text: str, **kwargs: Any) -> List[float]:
        """Asynchronous Embed query text."""
        embeddings = await self.aembed_documents([text], **kwargs)
        return embeddings[0]
