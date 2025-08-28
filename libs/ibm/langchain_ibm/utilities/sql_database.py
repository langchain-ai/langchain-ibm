import urllib.parse
from typing import Any, Dict, Iterable, List, Optional, Union

try:
    import pyarrow.flight as flight  # type: ignore[import-untyped]
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "To use WatsonxSQLDatabase one need to install langchain-ibm with extras "
        "`sql_toolkit`: `pip install langchain-ibm[sql_toolkit]`"
    ) from e

from ibm_watsonx_ai import APIClient, Credentials  # type: ignore[import-untyped]
from ibm_watsonx_ai.helpers.connections.flight_sql_service import (  # type: ignore[import-untyped]
    FlightSQLClient,
)
from langchain_core.utils.utils import from_env


def truncate_word(content: Any, *, length: int, suffix: str = "...") -> str:
    """
    Truncate a string to a certain number of characters, based on the max string
    length.

    Based on the analogous function from langchain_common.utilities.sql_database.py
    """

    if not isinstance(content, str) or length <= 0:
        return content

    if len(content) <= length:
        return content

    return content[: length - len(suffix)].rsplit(" ", 1)[0] + suffix


def _validate_param(value: Optional[str], key: str, env_key: str) -> None:
    if value is None:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )
    return None


def _from_env(env_var_name: str) -> Optional[str]:
    """Read env variable. If it is not set, return None."""
    return from_env(env_var_name, default=None)()


def pretty_print_table_info(schema: str, table_name: str, table_info: dict) -> str:
    def convert_column_data(field_metadata: dict) -> str:
        name = field_metadata.get("name")

        field_metadata_type = field_metadata.get("type", {})
        native_type = field_metadata_type.get("native_type")
        nullable = field_metadata_type.get("nullable")

        return f"{name} {native_type}{'' if nullable else ' NOT NULL'},"

    create_table_template = """
CREATE TABLE {schema}.{table_name} (
\t{column_definitions}{primary_key}
\t)"""

    primary_key: dict = next(
        filter(
            lambda el: el.get("name") == "primary_key",
            table_info.get("extended_metadata", [{}]),
        ),
        {},
    )
    key_columns = primary_key.get("value", {}).get("key_columns", [])

    return create_table_template.format(
        schema=schema,
        table_name=table_name,
        column_definitions="\n\t".join(
            [
                convert_column_data(field_metadata=field_metadata)
                for field_metadata in table_info["fields"]
            ]
        ),
        primary_key=f"\n\tPRIMARY KEY ({', '.join(key_columns)})"
        if primary_key
        else "",
    )


class WatsonxSQLDatabase:
    """Watsonx SQL Database class for IBM watsonx.ai databases
    connection assets. Uses Arrow Flight to interact with databases via watsonx.

    :param connection_id: ID of db connection asset
    :type connection_id: str

    :param schema: name of the database schema from which tables will be read
    :type schema: str

    :param project_id: ID of project, defaults to None
    :type project_id: Optional[str], optional

    :param space_id: ID of space, defaults to None
    :type space_id: Optional[str], optional

    :param url: URL to the Watson Machine Learning or CPD instance, defaults to None
    :type url: Optional[str], optional

    :param apikey: API key to the Watson Machine Learning
                   or CPD instance, defaults to None
    :type apikey: Optional[str], optional

    :param token: service token, used in token authentication, defaults to None
    :type token: Optional[str], optional

    :param password: password to the CPD instance., defaults to None
    :type password: Optional[str], optional

    :param username: username to the CPD instance., defaults to None
    :type username: Optional[str], optional

    :param instance_id: instance_id of the CPD instance., defaults to None
    :type instance_id: Optional[str], optional

    :param version: version of the CPD instance, defaults to None
    :type version: Optional[str], optional

    :param verify: certificate verification flag, defaults to None
    :type verify: Union[str, bool, None], optional

    :param watsonx_client: instance of `ibm_watsonx_ai.APIClient`, defaults to None
    :type watsonx_client: Optional[APIClient], optional

    :param ignore_tables: list of tables that will be ignored, defaults to None
    :type ignore_tables: Optional[List[str]], optional

    :param include_tables: list of tables that should be included, defaults to None
    :type include_tables: Optional[List[str]], optional

    :param sample_rows_in_table_info: number of first rows to be added to the
                                     table info, defaults to 3
    :type sample_rows_in_table_info: int, optional

    :param max_string_length: max length of string, defaults to 300
    :type max_string_length: int, optional

    :raises ValueError: raise if some required credentials are missing
    :raises RuntimeError: raise if no tables found in given schema
    """

    def __init__(
        self,
        *,
        connection_id: str,
        schema: str,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        url: Optional[str] = None,
        apikey: Optional[str] = None,
        token: Optional[str] = None,
        password: Optional[str] = None,
        username: Optional[str] = None,
        instance_id: Optional[str] = None,
        version: Optional[str] = None,
        verify: Union[str, bool, None] = None,
        watsonx_client: Optional[APIClient] = None,  #: :meta private:
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3,
        max_string_length: int = 300,
    ) -> None:
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

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
                if not token and not apikey:
                    raise ValueError(
                        "Did not find 'apikey' or 'token',"
                        " please add an environment variable"
                        " `WATSONX_APIKEY` or 'WATSONX_TOKEN' "
                        "which contains it,"
                        " or pass 'apikey' or 'token'"
                        " as a named parameter."
                    )
            else:
                token = token or _from_env("WATSONX_TOKEN")
                apikey = apikey or _from_env("WATSONX_APIKEY")
                password = password or _from_env("WATSONX_PASSWORD")
                if not token and not password and not apikey:
                    raise ValueError(
                        "Did not find 'token', 'password' or 'apikey',"
                        " please add an environment variable"
                        " `WATSONX_TOKEN`, 'WATSONX_PASSWORD' or 'WATSONX_APIKEY' "
                        "which contains it,"
                        " or pass 'token', 'password' or 'apikey'"
                        " as a named parameter."
                    )

                try:
                    _validate_param(token, "token", "WATSONX_TOKEN")
                except ValueError:
                    pass

                def _check_with_username(
                    auth_name: str, auth_object: Any, username: Optional[str] = username
                ) -> None:
                    try:
                        _validate_param(
                            auth_object, auth_name, "WATSONX_" + auth_name.upper()
                        )
                    except ValueError:
                        pass
                    else:
                        username = username or _from_env("WATSONX_USERNAME")
                        _validate_param(username, "username", "WATSONX_USERNAME")

                # validate if password is passed then username is also passed
                _check_with_username(auth_name="password", auth_object=password)

                # validate if apikey is passed then username is also passed
                _check_with_username(auth_name="apikey", auth_object=apikey)

                instance_id = instance_id or _from_env("WATSONX_INSTANCE_ID")
                _validate_param(instance_id, "instance_id", "WATSONX_INSTANCE_ID")

            credentials = Credentials(
                url=url,
                api_key=apikey,
                token=token,
                password=password,
                username=username,
                instance_id=instance_id,
                version=version,
                verify=verify,
            )
            self.watsonx_client = APIClient(
                credentials=credentials, project_id=project_id, space_id=space_id
            )
            project_id = project_id or _from_env("WATSONX_PROJECT_ID")
            space_id = space_id or _from_env("WATSONX_SPACE_ID")
            if project_id:
                self.watsonx_client.set.default_project(project_id=project_id)
            elif space_id:
                self.watsonx_client.set.default_space(space_id=space_id)

        else:
            self.watsonx_client = watsonx_client

        context_id: dict[str, Optional[str]] = {"project_id": None, "space_id": None}
        if project_id is not None:
            context_id["project_id"] = project_id
        elif space_id is not None:
            context_id["space_id"] = space_id
        elif self.watsonx_client.default_project_id is not None:
            context_id["project_id"] = self.watsonx_client.default_project_id
        elif self.watsonx_client.default_space_id is not None:
            context_id["space_id"] = self.watsonx_client.default_space_id
        else:
            raise ValueError("Either project_id or space_id is required.")

        self._flight_sql_client = FlightSQLClient(
            connection_id=connection_id, api_client=self.watsonx_client, **context_id
        )

        with self._flight_sql_client as flight_sql_client:
            _tables = flight_sql_client.get_tables(schema=self.schema).get("assets")
            if _tables is not None:
                self._all_tables = {
                    table.get("name") for table in _tables if table.get("name")
                }
            else:
                raise RuntimeError(f"No tables found in the schema: {schema}")

            if self._include_tables:
                missing_tables = self._include_tables - self._all_tables
                if missing_tables:
                    raise ValueError(
                        f"include_tables {missing_tables} not found in database"
                    )
            if self._ignore_tables:
                missing_tables = self._ignore_tables - self._all_tables
                if missing_tables:
                    raise ValueError(
                        f"ignore_tables {missing_tables} not found in database"
                    )

            self._meta_all_tables = {
                table_name: flight_sql_client.get_table_info(
                    table_name=table_name, schema=self.schema
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
    ) -> dict:
        """Execute a command."""

        with self._flight_sql_client as flight_sql_client:
            results = flight_sql_client.execute(query=command)

        return results.to_dict("records")

    def run(self, command: str, include_columns: bool = False) -> str:
        """Execute a SQL command and return a string representing the results."""
        result = self._execute(command)

        res: List[Dict] = [
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

    def get_table_info(self, table_names: Optional[Iterable[str]] = None) -> str:
        """Get information about specified tables."""

        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")

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
                    ).to_string()
                    for table_name in table_names
                ]
            )

    def get_table_info_no_throw(
        self, table_names: Optional[Iterable[str]] = None
    ) -> str:
        """Get information about specified tables."""
        try:
            return self.get_table_info(table_names=table_names)
        except (flight.FlightError, ValueError) as e:
            """Format the error message"""
            return f"Error: {e}"

    def get_context(self) -> Dict[str, Any]:
        """Return db context that you may want in agent prompt."""
        table_names = list(self.get_usable_table_names())
        table_info = self.get_table_info_no_throw()
        return {"table_info": table_info, "table_names": ", ".join(table_names)}
