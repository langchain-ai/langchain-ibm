import logging
from typing import Any, Dict, List, Optional, Union

from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings  # type: ignore
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from langchain_core.utils.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_ibm.utils import check_for_attribute, extract_params

logger = logging.getLogger(__name__)


class WatsonxEmbeddings(BaseModel, LangChainEmbeddings):
    """IBM watsonx.ai embedding models."""

    model_id: str
    """Type of model to use."""

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
        * False - no verification will be made"""

    watsonx_embed: Embeddings = Field(default=None)  #: :meta private:

    watsonx_client: Optional[APIClient] = Field(default=None)  #: :meta private:

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that credentials and python package exists in environment."""
        if isinstance(self.watsonx_client, APIClient):
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

            watsonx_embed = Embeddings(
                model_id=self.model_id,
                params=self.params,
                credentials=credentials,
                project_id=self.project_id,
                space_id=self.space_id,
            )

            self.watsonx_embed = watsonx_embed

        return self

    def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Embed search docs."""
        params = extract_params(kwargs, self.params)
        return self.watsonx_embed.embed_documents(
            texts=texts, **(kwargs | {"params": params})
        )

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text], **kwargs)[0]
