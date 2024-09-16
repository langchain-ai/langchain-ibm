import os
from typing import Dict, List, Optional, Union

from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings  # type: ignore
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Extra,
    Field,
    SecretStr,
    model_validator,
    root_validator,
)
from typing_extensions import Self


class WatsonxEmbeddings(BaseModel, LangChainEmbeddings):
    """IBM WatsonX.ai embedding models."""

    model_id: str = ""
    """Type of model to use."""

    project_id: str = ""
    """ID of the Watson Studio project."""

    space_id: str = ""
    """ID of the Watson Studio space."""

    url: Optional[SecretStr] = None
    """Url to Watson Machine Learning or CPD instance"""

    apikey: Optional[SecretStr] = None
    """Apikey to Watson Machine Learning or CPD instance"""

    token: Optional[SecretStr] = None
    """Token to CPD instance"""

    password: Optional[SecretStr] = None
    """Password to CPD instance"""

    username: Optional[SecretStr] = None
    """Username to CPD instance"""

    instance_id: Optional[SecretStr] = None
    """Instance_id of CPD instance"""

    version: Optional[SecretStr] = None
    """Version of CPD instance"""

    params: Optional[dict] = None
    """Model parameters to use during generate requests."""

    verify: Union[str, bool, None] = None
    """User can pass as verify one of following:
        the path to a CA_BUNDLE file
        the path of directory with certificates of trusted CAs
        True - default path to truststore will be taken
        False - no verification will be made"""

    watsonx_embed: Embeddings = Field(default=None)  #: :meta private:

    watsonx_client: APIClient = Field(default=None)  #: :meta private:

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
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
            self.url = convert_to_secret_str(
                get_from_dict_or_env(values, "url", "WATSONX_URL")
            )
            if "cloud.ibm.com" in self.url.get_secret_value():
                self.apikey = convert_to_secret_str(
                    get_from_dict_or_env(values, "apikey", "WATSONX_APIKEY")
                )
            else:
                if (
                    not self.token
                    and "WATSONX_TOKEN" not in os.environ
                    and not self.password
                    and "WATSONX_PASSWORD" not in os.environ
                    and not self.apikey
                    and "WATSONX_APIKEY" not in os.environ
                ):
                    raise ValueError(
                        "Did not find 'token', 'password' or 'apikey',"
                        " please add an environment variable"
                        " `WATSONX_TOKEN`, 'WATSONX_PASSWORD' or 'WATSONX_APIKEY' "
                        "which contains it,"
                        " or pass 'token', 'password' or 'apikey'"
                        " as a named parameter."
                    )
                elif self.token or "WATSONX_TOKEN" in os.environ:
                    self.token = convert_to_secret_str(
                        get_from_dict_or_env(values, "token", "WATSONX_TOKEN")
                    )
                elif self.password or "WATSONX_PASSWORD" in os.environ:
                    self.password = convert_to_secret_str(
                        get_from_dict_or_env(values, "password", "WATSONX_PASSWORD")
                    )
                    self.username = convert_to_secret_str(
                        get_from_dict_or_env(values, "username", "WATSONX_USERNAME")
                    )
                elif self.apikey or "WATSONX_APIKEY" in os.environ:
                    self.apikey = convert_to_secret_str(
                        get_from_dict_or_env(values, "apikey", "WATSONX_APIKEY")
                    )
                    self.username = convert_to_secret_str(
                        get_from_dict_or_env(values, "username", "WATSONX_USERNAME")
                    )
                if not self.instance_id or "WATSONX_INSTANCE_ID" not in os.environ:
                    self.instance_id = convert_to_secret_str(
                        get_from_dict_or_env(
                            values, "instance_id", "WATSONX_INSTANCE_ID"
                        )
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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.watsonx_embed.embed_documents(texts=texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
