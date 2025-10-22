"""IBM watsonx.ai rerank wrapper."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from ibm_watsonx_ai import APIClient  # type: ignore[import-untyped]
from ibm_watsonx_ai.foundation_models import Rerank  # type: ignore[import-untyped]
from ibm_watsonx_ai.foundation_models.schema import (  # type: ignore[import-untyped]
    RerankParameters,
)
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils.utils import secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_ibm.utils import extract_params, resolve_watsonx_credentials

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.callbacks import Callbacks


class WatsonxRerank(BaseDocumentCompressor):
    """Document compressor that uses `watsonx Rerank API`.

    ???+ info "Setup"

        To use, you should have `langchain_ibm` python package installed,
        and the environment variable `WATSONX_APIKEY` set with your API key, or pass
        it as a named parameter `apikey` to the constructor.

        ```bash
        pip install -U langchain-ibm

        # or using uv
        uv add langchain-ibm
        ```

        ```bash
        export WATSONX_APIKEY="your-api-key"
        ```

    ??? info "Instantiate"

        ```python
        from langchain_ibm import WatsonxRerank
        from ibm_watsonx_ai.foundation_models.schema import RerankParameters

        parameters = RerankParameters(truncate_input_tokens=20)

        ranker = WatsonxRerank(
            model_id="cross-encoder/ms-marco-minilm-l-12-v2",
            url="https://us-south.ml.cloud.ibm.com",
            project_id="*****",
            params=parameters,
            # apikey="*****"
        )
        ```

    ??? info "Rerank"

        ```python
        query = "red cat chasing a laser pointer"
        documents = [
            "A red cat darts across the living room, pouncing on a red laser dot.",
            "Two dogs play fetch in the park with a tennis ball.",
            "The tabby cat naps on a sunny windowsill all afternoon.",
            "A recipe for tuna casserole with crispy breadcrumbs.",
        ]

        ranker.rerank(documents=documents, query=query)
        ```

        ```python
        [
            {"index": 0, "relevance_score": 0.8719543218612671},
            {"index": 2, "relevance_score": 0.6520894169807434},
            {"index": 1, "relevance_score": 0.6270776391029358},
            {"index": 3, "relevance_score": 0.4607713520526886},
        ]
        ```

    """

    model_id: str
    """Type of model to use."""

    project_id: str | None = None
    """ID of the Watson Studio project."""

    space_id: str | None = None
    """ID of the Watson Studio space."""

    url: SecretStr = Field(
        alias="url",
        default_factory=secret_from_env("WATSONX_URL", default=None),  # type: ignore[assignment]
    )
    """URL to the Watson Machine Learning or CPD instance."""

    apikey: SecretStr | None = Field(
        alias="apikey",
        default_factory=secret_from_env("WATSONX_APIKEY", default=None),
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

    params: dict | RerankParameters | None = None
    """Model parameters to use during request generation."""

    verify: str | bool | None = None
    """You can pass one of following as verify:
        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * True - default path to truststore will be taken
        * False - no verification will be made"""

    validate_model: bool = True
    """Model ID validation."""

    streaming: bool = False
    """ Whether to stream the results or not. """

    watsonx_rerank: Rerank = Field(default=None, exclude=True)  #: :meta private:

    watsonx_client: APIClient | None = Field(default=None, exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        protected_namespaces=(),
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Mapping of secret environment variables."""
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
            watsonx_rerank = Rerank(
                model_id=self.model_id,
                params=self.params,
                api_client=self.watsonx_client,
                project_id=self.project_id,
                space_id=self.space_id,
                verify=self.verify,
            )
            self.watsonx_rerank = watsonx_rerank

        else:
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

            watsonx_rerank = Rerank(
                model_id=self.model_id,
                credentials=credentials,
                params=self.params,
                project_id=self.project_id,
                space_id=self.space_id,
                verify=self.verify,
            )
            self.watsonx_rerank = watsonx_rerank

        return self

    def rerank(
        self,
        documents: Sequence[str | Document | dict],
        query: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Rerank documents."""
        if len(documents) == 0:  # to avoid empty api call
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        params = extract_params(kwargs, self.params)

        results = self.watsonx_rerank.generate(
            query=query,
            inputs=docs,
            **(kwargs | {"params": params}),
        )
        return [
            {"index": res.get("index"), "relevance_score": res.get("score")}
            for res in results["results"]
        ]

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
        **kwargs: Any,
    ) -> Sequence[Document]:
        """Compress documents using watsonx's rerank API.

        Args:
            documents: A sequence of documents to compress
            query: The query to use for compressing the documents
            callbacks: Callbacks to run during the compression process
            kwargs: Additional keyword args

        Returns:
            A sequence of compressed documents
        """
        compressed = []
        for res in self.rerank(documents, query, **kwargs):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed
