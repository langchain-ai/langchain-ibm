"""Utilities for watsonx SQL integration."""

import contextlib
import urllib.parse
from collections.abc import Iterable
from typing import Any

try:
    from pyarrow import flight  # type: ignore[import-untyped]
except ModuleNotFoundError as e:
    error_msg = (
        "To use WatsonxSQLDatabase one need to install langchain-ibm with extras "
        "`sql_toolkit`: `pip install langchain-ibm[sql_toolkit]`",
    )
    raise ModuleNotFoundError(error_msg) from e

from ibm_watsonx_ai import APIClient, Credentials  # type: ignore[import-untyped]
from ibm_watsonx_ai.helpers.connections.flight_sql_service import (  # type: ignore[import-untyped]
    FlightSQLClient,
)
from langchain_core.utils.utils import from_env


def truncate_word(content: Any, *, length: int, suffix: str = "...") -> str | Any:
    """Truncate a string to a maximum length.

    Based on the analogous function from langchain_common.utilities.sql_database.py.
    """
    if not isinstance(content, str) or length <= 0:
        return content

    if len(content) <= length:
        return content

    return content[: length - len(suffix)].rsplit(" ", 1)[0] + suffix


def _validate_param(value: str | None, key: str, env_key: str) -> None:
    if value is None:
        error_msg = (
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )
        raise ValueError(error_msg)


def _from_env(env_var_name: str) -> str | None:
    """Read env variable. If it is not set, return None."""
    return from_env(env_var_name, default=None)()


def pretty_print_table_info(
    schema: str, table_name: str, table_info: dict[str, Any]
) -> str:
    """Pretty print table info."""

    def convert_column_data(field_metadata: dict[str, Any]) -> str:
        name = field_metadata.get("name")

        field_metadata_type = field_metadata.get("type", {})
        native_type = field_metadata_type.get("native_type")
        nullable = field_metadata_type.get("nullable")

        return f'"{name}" {native_type}{"" if nullable else " NOT NULL"}'

    create_table_template = """
CREATE TABLE "{schema}"."{table_name}" (
\t{column_definitions}{primary_key}{foreign_key}
\t)"""

    extended_metadata = table_info.get("extended_metadata", [{}])

    def _retrieve_field_data(field_name: str) -> dict[str, Any]:
        return next(
            filter(
                lambda el: el.get("name") == field_name,
                extended_metadata,
            ),
            {},
        )

    # Primary Key
    primary_key: dict[str, Any] = _retrieve_field_data("primary_key")
    if primary_key:
        key_columns = ", ".join(primary_key.get("value", {}).get("key_columns", []))
        primary_key_text = (
            f",\n\tCONSTRAINT {primary_key['name']} PRIMARY KEY ({key_columns})"
        )
    else:
        primary_key_text = ""

    # Foreign keys
    foreign_keys: dict[str, Any] = _retrieve_field_data("foreign_keys")
    if foreign_keys:
        foreign_keys_text = ""
        foreign_key_text_template = (
            "CONSTRAINT {fk_name} FOREIGN KEY ({col_name}) "
            "REFERENCES {external_table_name}({external_col_name})"
        )
        for foreign_key in foreign_keys.get("value", []):
            foreign_keys_text += ",\n\t"
            join_condition = foreign_key["join_condition"].split("=")
            foreign_keys_text += foreign_key_text_template.format(
                fk_name=foreign_key["name"],
                col_name=join_condition[0].strip().split(".")[-1],
                external_table_name=join_condition[1].strip().split(".")[1],
                external_col_name=join_condition[1].strip().split(".")[2],
            )
    else:
        foreign_keys_text = ""

    return create_table_template.format(
        schema=schema,
        table_name=table_name,
        column_definitions="\n\t".join(
            [
                # Do not add comma for the last column
                convert_column_data(field_metadata=field_metadata) + ","
                if index < len(table_info["fields"])
                else convert_column_data(field_metadata=field_metadata)
                for index, field_metadata in enumerate(table_info["fields"], start=1)
            ],
        ),
        primary_key=primary_key_text,
        foreign_key=foreign_keys_text,
    )


class WatsonxSQLDatabase:
    """Watsonx SQL Database class for IBM watsonx.ai databases connection assets.

    Uses Arrow Flight to interact with databases via watsonx.

    Args:
        connection_id: ID of db connection asset
        schema: name of the database schema from which tables will be read
        project_id: ID of project
        space_id: ID of space
        url: URL to the Watson Machine Learning or CPD instance
        api_key: API key to the Watson Machine Learning or CPD instance
        apikey: API key to the Watson Machine Learning or CPD instance (deprecated)
        token: service token, used in token authentication
        password: password to the CPD instance
        username: username to the CPD instance
        instance_id: instance_id of the CPD instance
        version: version of the CPD instance
        verify: certificate verification flag
        watsonx_client: instance of `ibm_watsonx_ai.APIClient`
        ignore_tables: list of tables that will be ignored
        include_tables: list of tables that should be included
        sample_rows_in_table_info: number of first rows to be added to the table info
        max_string_length: max length of string


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
        from langchain_ibm.utilities.sql_database import WatsonxSQLDatabase

        wx_sql_database = WatsonxSQLDatabase(
            connection_id="<CONNECTION_ID>",
            schema="<SCHEMA>",
            url="<URL>",
            project_id="<PROJECT_ID>",
            api_key="<API_KEY>",
        )
        ```

    ??? warning "Raises"
        - `ValueError` - if some required credentials are missing
        - `RuntimeError` - if no tables found in given schema

    """

    def __init__(
        self,
        *,
        connection_id: str,
        schema: str,
        project_id: str | None = None,
        space_id: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        apikey: str | None = None,
        token: str | None = None,
        password: str | None = None,
        username: str | None = None,
        instance_id: str | None = None,
        version: str | None = None,
        verify: str | bool | None = None,
        watsonx_client: APIClient | None = None,  #: :meta private:
        ignore_tables: list[str] | None = None,
        include_tables: list[str] | None = None,
        sample_rows_in_table_info: int = 3,
        max_string_length: int = 300,
    ) -> None:
        """WatsonxSQLDatabase class."""
        if include_tables and ignore_tables:
            error_msg = "Cannot specify both include_tables and ignore_tables"
            raise ValueError(error_msg)

        self.schema = schema
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        self._include_tables = set(include_tables) if include_tables else set()
        self._sample_rows_in_table_info = sample_rows_in_table_info

        self._max_string_length = max_string_length

        if watsonx_client is None:
            url = url or _from_env("WATSONX_URL")
            _validate_param(url, "url", "WATSONX_URL")

            parsed_url = urllib.parse.urlparse(url)
            if parsed_url.netloc.endswith(".cloud.ibm.com"):  # type: ignore[arg-type]
                token = token or _from_env("WATSONX_TOKEN")
                apikey = apikey or _from_env("WATSONX_APIKEY")
                api_key = api_key or _from_env("WATSONX_API_KEY")
                key_to_use = api_key or apikey
                if not token and not key_to_use:
                    error_msg = (
                        "Did not find 'api_key' or 'token',"
                        " please add an environment variable"
                        " `WATSONX_API_KEY` or 'WATSONX_TOKEN' "
                        "which contains it,"
                        " or pass 'api_key' or 'token'"
                        " as a named parameter."
                    )
                    raise ValueError(error_msg)
            else:
                token = token or _from_env("WATSONX_TOKEN")
                apikey = apikey or _from_env("WATSONX_APIKEY")
                api_key = api_key or _from_env("WATSONX_API_KEY")
                key_to_use = api_key or apikey
                password = password or _from_env("WATSONX_PASSWORD")
                if not token and not password and not key_to_use:
                    error_msg = (
                        "Did not find 'token', 'password' or 'api_key',"
                        " please add an environment variable"
                        " `WATSONX_TOKEN`, 'WATSONX_PASSWORD' or 'WATSONX_API_KEY' "
                        "which contains it,"
                        " or pass 'token', 'password' or 'api_key'"
                        " as a named parameter."
                    )
                    raise ValueError(error_msg)

                with contextlib.suppress(ValueError):
                    _validate_param(token, "token", "WATSONX_TOKEN")

                def _check_with_username(
                    auth_name: str,
                    auth_object: Any,
                    username: str | None = username,
                ) -> None:
                    try:
                        _validate_param(
                            auth_object,
                            auth_name,
                            "WATSONX_" + auth_name.upper(),
                        )
                    except ValueError:
                        pass
                    else:
                        username = username or _from_env("WATSONX_USERNAME")
                        _validate_param(username, "username", "WATSONX_USERNAME")

                # validate if password is passed then username is also passed
                _check_with_username(auth_name="password", auth_object=password)

                # validate if api_key is passed then username is also passed
                _check_with_username(auth_name="api_key", auth_object=key_to_use)

                instance_id = instance_id or _from_env("WATSONX_INSTANCE_ID")
                _validate_param(instance_id, "instance_id", "WATSONX_INSTANCE_ID")

            credentials = Credentials(
                url=url,
                api_key=key_to_use,
                token=token,
                password=password,
                username=username,
                instance_id=instance_id,
                version=version,
                verify=verify,
            )
            self.watsonx_client = APIClient(
                credentials=credentials,
                project_id=project_id,
                space_id=space_id,
            )
            project_id = project_id or _from_env("WATSONX_PROJECT_ID")
            space_id = space_id or _from_env("WATSONX_SPACE_ID")
            if project_id:
                self.watsonx_client.set.default_project(project_id=project_id)
            elif space_id:
                self.watsonx_client.set.default_space(space_id=space_id)

        else:
            self.watsonx_client = watsonx_client

        context_id: dict[str, str | None] = {"project_id": None, "space_id": None}
        if project_id is not None:
            context_id["project_id"] = project_id
        elif space_id is not None:
            context_id["space_id"] = space_id
        elif self.watsonx_client.default_project_id is not None:
            context_id["project_id"] = self.watsonx_client.default_project_id
        elif self.watsonx_client.default_space_id is not None:
            context_id["space_id"] = self.watsonx_client.default_space_id
        else:
            error_msg = "Either project_id or space_id is required."
            raise ValueError(error_msg)

        self._flight_sql_client = FlightSQLClient(
            connection_id=connection_id,
            api_client=self.watsonx_client,
            **context_id,
        )

        with self._flight_sql_client as flight_sql_client:
            _tables = flight_sql_client.get_tables(schema=self.schema).get("assets")
            if _tables is not None:
                self._all_tables = {
                    table.get("name") for table in _tables if table.get("name")
                }
            else:
                error_msg = f"No tables found in the schema: {schema}"
                raise RuntimeError(error_msg)

            if self._include_tables:
                missing_tables = self._include_tables - self._all_tables
                if missing_tables:
                    error_msg = f"include_tables {missing_tables} not found in database"
                    raise ValueError(error_msg)
            if self._ignore_tables:
                missing_tables = self._ignore_tables - self._all_tables
                if missing_tables:
                    error_msg = f"ignore_tables {missing_tables} not found in database"
                    raise ValueError(error_msg)

            self._meta_all_tables = {
                table_name: flight_sql_client.get_table_info(
                    table_name=table_name,
                    schema=self.schema,
                    extended_metadata=True,
                    interaction_properties=True,
                )
                for table_name in self._all_tables
                if table_name in (self._include_tables or self._all_tables)
                and table_name not in (self._ignore_tables or {})
            }

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return sorted(self._include_tables)
        return sorted(self._all_tables - self._ignore_tables)

    def _execute(
        self,
        command: str,
    ) -> Any:
        """Execute a command."""
        with self._flight_sql_client as flight_sql_client:
            results = flight_sql_client.execute(query=command)

        return results.to_dict("records")

    def run(self, command: str, *, include_columns: bool = False) -> str:
        """Execute a SQL command and return a string representing the results."""
        result = self._execute(command)

        res: list[dict[str, Any]] = [
            {
                column: truncate_word(value, length=self._max_string_length)
                for column, value in r.items()
            }
            for r in result
        ]

        if not include_columns:
            res = [tuple(row.values()) for row in res]  # type: ignore[misc]

        return str(res) if res else ""

    def run_no_throw(
        self,
        command: str,
        *,
        include_columns: bool = False,
    ) -> str:
        """Execute a SQL command and return a string representing the results.

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

    def get_table_info(self, table_names: Iterable[str] | None = None) -> str:
        """Get information about specified tables."""
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                error_msg = f"table_names {missing_tables} not found in database"
                raise ValueError(error_msg)

        with self._flight_sql_client as flight_sql_client:
            if table_names is None:
                table_names = self._all_tables

            return "\n\n".join(
                [
                    pretty_print_table_info(
                        schema=self.schema,
                        table_name=table_name,
                        table_info=self._meta_all_tables[table_name],
                    )
                    + f"\n\nFirst {self._sample_rows_in_table_info} rows "
                    + f"of table {table_name}:\n\n"
                    + flight_sql_client.get_n_first_rows(
                        schema=self.schema,
                        table_name=table_name,
                        n=self._sample_rows_in_table_info,
                    ).to_string(index=False)
                    for table_name in table_names
                ],
            )

    def get_table_info_no_throw(self, table_names: Iterable[str] | None = None) -> str:
        """Get information about specified tables."""
        try:
            return self.get_table_info(table_names=table_names)
        except (flight.FlightError, ValueError) as e:
            """Format the error message"""
            return f"Error: {e}"

    def get_context(self) -> dict[str, Any]:
        """Return db context that you may want in agent prompt."""
        table_names = list(self.get_usable_table_names())
        table_info = self.get_table_info_no_throw()
        return {"table_info": table_info, "table_names": ", ".join(table_names)}
