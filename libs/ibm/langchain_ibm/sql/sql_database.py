import json
from typing import Any, Dict, Iterable, List, Optional

import pyarrow.flight as flight
from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
from ibm_watsonx_ai.helpers.connections.flight_sql_service import FlightSQLClient


def truncate_word(content: Any, *, length: int, suffix: str = "...") -> str:
    """
    Truncate a string to a certain number of words, based on the max string
    length.

    Based on the analogous function from langchain_common.utilities.sql_database.py
    """

    if not isinstance(content, str) or length <= 0:
        return content

    if len(content) <= length:
        return content

    return content[: length - len(suffix)].rsplit(" ", 1)[0] + suffix


class WatsonxSQLDatabase:
    """Watsonx SQL Database class for IBM watsonx.ai connection assets.

    Support view by default."""

    def __init__(
        self,
        *,
        connection_id: str,
        schema: str,
        project_id: str | None = None,
        space_id: str | None = None,
        url: str | None = None,
        apikey: str | None = None,
        token: str | None = None,
        password: str | None = None,
        username: str | None = None,
        instance_id: str | None = None,
        version: str | None = None,
        watsonx_client: Optional[APIClient] = None,  #: :meta private:
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3,
        max_string_length: int = 300,
    ) -> None:
        if watsonx_client is None:
            self.watsonx_client = APIClient(
                credentials=Credentials(
                    api_key=apikey,
                    token=token,
                    url=url,
                    username=username,
                    password=password,
                    instance_id=instance_id,
                    version=version,
                ),
                project_id=project_id,
                space_id=space_id,
            )
        else:
            self.watsonx_client = watsonx_client

        context_id: dict[str, str | None] = {"project_id": None, "space_id": None}
        if project_id is not None:
            context_id["project_id"] = project_id
        elif space_id is not None:
            context_id["space_id"] = space_id
        elif watsonx_client.default_project_id is not None:
            context_id["project_id"] = watsonx_client.default_project_id
        elif watsonx_client.default_space_id is not None:
            context_id["space_id"] = watsonx_client.default_space_id
        else:
            raise ValueError("Either project_id or space_id is required.")

        self._flight_sql_client = FlightSQLClient(
            connection_id=connection_id, api_client=self.watsonx_client, **context_id
        )

        self.schema = schema
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        self._include_tables = set(include_tables) if include_tables else set()
        self._sample_rows_in_table_info = sample_rows_in_table_info

        self._max_string_length = max_string_length

        # caching some basic info
        # TODO: use open connection

        with self._flight_sql_client as flight_sql_client:
            self._all_tables = {
                table.get("name")
                for table in flight_sql_client.get_tables(schema=self.schema)["assets"]
            }

            self._meta_all_tables = {
                table_name: flight_sql_client.get_table_info(
                    table_name=table_name, schema=self.schema
                )
                for table_name in self._all_tables
            }

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return sorted(self._include_tables)
        return sorted(self._all_tables - self._ignore_tables)

    def _execute(
        self,
        command: str,
    ) -> dict:
        """Execute a command."""

        # TODO: set schema in query

        with self._flight_sql_client as flight_sql_client:
            results = flight_sql_client.execute(query=command)

        return results.to_dict("records")

    def run(self, command: str, include_columns: bool = False) -> str:
        """Execute a SQL command and return a string representing the results."""
        # TODO: check if to_string not be better
        result = self._execute(command)

        res = [
            {
                column: truncate_word(value, length=self._max_string_length)
                for column, value in r.items()
            }
            for r in result
        ]

        if not include_columns:
            res = [tuple(row.values()) for row in res]

        return str(res) if res else ""

    def run_no_throw(
        self,
        command: str,
        include_columns: bool = False,
    ) -> str:
        """Execute a SQL command and return a string representing the results

        If the statement throws an error, the error message is returned.
        """
        try:
            return self.run(
                command,
                include_columns=include_columns,
            )
        except flight.FlightError as e:
            """Format the error message"""
            return f"Error: {e}"

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables."""

        extra_interaction_properties = {
            "infer_schema": "true",
            "row_limit": self._sample_rows_in_table_info,
        }

        if table_names is not None:
            with self._flight_sql_client as flight_sql_client:
                return "\n\n".join(
                    [
                        json.dumps(self._meta_all_tables[table_name], indent=2)
                        + f"\n\n First {self._sample_rows_in_table_info} rows of table {table_name}:\n\n"
                        + flight_sql_client.execute(
                            None,
                            interaction_properties=extra_interaction_properties
                            | {"table_name": table_name},
                        ).to_string()
                        for table_name in table_names
                    ]
                )
        else:
            return "\n\n".join(self._meta_all_tables)

    def get_table_info_no_throw(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables."""
        try:
            return self.get_table_info(table_names=table_names)
        except flight.FlightError as e:
            """Format the error message"""
            return f"Error: {e}"

    def get_context(self) -> Dict[str, Any]:
        """Return db context that you may want in agent prompt."""
        table_names = list(self.get_usable_table_names())
        table_info = self.get_table_info_no_throw()
        return {"table_info": table_info, "table_names": ", ".join(table_names)}
