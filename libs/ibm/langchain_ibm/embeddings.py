import os
from typing import Any, Dict, List, Optional, Union

from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings  # type: ignore
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)


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
        protected_namespaces=(),
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that credentials and python package exists in environment."""
        if isinstance(values.get("watsonx_client"), APIClient):
            watsonx_embed = Embeddings(
                model_id=values.get("model_id", ""),
                params=values.get("params"),
                api_client=values.get("watsonx_client"),
                project_id=values.get("project_id", ""),
                space_id=values.get("space_id", ""),
                verify=values.get("verify"),
            )
            values["watsonx_embed"] = watsonx_embed

        else:
            values["url"] = convert_to_secret_str(
                get_from_dict_or_env(values, "url", "WATSONX_URL")
            )
            if "cloud.ibm.com" in values.get("url", "").get_secret_value():
                values["apikey"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "apikey", "WATSONX_APIKEY")
                )
            else:
                if (
                    not values.get("token")
                    and "WATSONX_TOKEN" not in os.environ
                    and not values.get("password")
                    and "WATSONX_PASSWORD" not in os.environ
                    and not values.get("apikey")
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
                elif values.get("token") or "WATSONX_TOKEN" in os.environ:
                    values["token"] = convert_to_secret_str(
                        get_from_dict_or_env(values, "token", "WATSONX_TOKEN")
                    )
                elif values.get("password") or "WATSONX_PASSWORD" in os.environ:
                    values["password"] = convert_to_secret_str(
                        get_from_dict_or_env(values, "password", "WATSONX_PASSWORD")
                    )
                    values["username"] = convert_to_secret_str(
                        get_from_dict_or_env(values, "username", "WATSONX_USERNAME")
                    )
                elif values.get("apikey") or "WATSONX_APIKEY" in os.environ:
                    values["apikey"] = convert_to_secret_str(
                        get_from_dict_or_env(values, "apikey", "WATSONX_APIKEY")
                    )
                    values["username"] = convert_to_secret_str(
                        get_from_dict_or_env(values, "username", "WATSONX_USERNAME")
                    )
                if (
                    not values.get("instance_id")
                    or "WATSONX_INSTANCE_ID" not in os.environ
                ):
                    values["instance_id"] = convert_to_secret_str(
                        get_from_dict_or_env(
                            values, "instance_id", "WATSONX_INSTANCE_ID"
                        )
                    )

            credentials = Credentials(
                url=values["url"].get_secret_value() if values.get("url") else None,
                api_key=values["apikey"].get_secret_value()
                if values.get("apikey")
                else None,
                token=values["token"].get_secret_value()
                if values.get("token")
                else None,
                password=values["password"].get_secret_value()
                if values.get("password")
                else None,
                username=values["username"].get_secret_value()
                if values.get("username")
                else None,
                instance_id=values["instance_id"].get_secret_value()
                if values.get("instance_id")
                else None,
                version=values["version"].get_secret_value()
                if values.get("version")
                else None,
                verify=values.get("verify"),
            )

            watsonx_embed = Embeddings(
                model_id=values.get("model_id", ""),
                params=values.get("params"),
                credentials=credentials,
                project_id=values.get("project_id", ""),
                space_id=values.get("space_id", ""),
            )

            values["watsonx_embed"] = watsonx_embed

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.watsonx_embed.embed_documents(texts=texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
