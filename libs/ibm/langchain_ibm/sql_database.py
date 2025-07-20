from typing import Any, Dict, List, Optional, Union

from ibm_watsonx_ai import APIClient, Credentials  # type: ignore

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

class WatsonxSQLDatabase:
    """Watsonx SQL Database class for IBM watsonx.ai connection assets.
    
    Support view by default."""

    def __init__(
        self,
        *,
        connection_id: str, 
        schema: str,
        project_id: str | None = None,
        span_id: str | None = None,
        url: str | None = None,
        apikey: str | None = None,
        token: str | None = None,
        password: str | None = None,
        username: str | None = None,
        instance_id: str | None = None,
        version: str | None = None,
        watsonx_client: Optional[APIClient] = Field(default=None),  #: :meta private:
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3,
        custom_table_info: Optional[dict] = None, # TODO: do we need it?
        max_string_length: int = 300,
    ) -> None:
        pass