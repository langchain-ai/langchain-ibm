import os
from collections.abc import Generator
from typing import Any
from unittest import mock
from unittest.mock import Mock, patch

import pandas as pd  # type: ignore[import-untyped]
import pytest
from pyarrow import flight  # type: ignore[import-untyped]

from langchain_ibm.utilities.sql_database import (
    WatsonxSQLDatabase,
    pretty_print_table_info,
    truncate_word,
)

CONNECTION_ID = "test_connection_id"
PROJECT_ID = "test_project_id"


@pytest.fixture
def schema() -> str:
    return "test_schema"


@pytest.fixture
def table_name() -> str:
    return "test_table"


@pytest.fixture
def table_info() -> dict:
    return {
        "fields": [
            {"name": "id", "type": {"native_type": "INT", "nullable": False}},
            {"name": "name", "type": {"native_type": "VARCHAR(255)", "nullable": True}},
            {"name": "age", "type": {"native_type": "INT", "nullable": True}},
        ],
        "extended_metadata": [
            {"name": "primary_key", "value": {"key_columns": ["id"]}}
        ],
    }


@pytest.fixture
def clear_env() -> Generator[None, None, None]:
    with mock.patch.dict(os.environ, clear=True):
        yield


class MockFlightSQLClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self, *args: Any, **kwargs: Any) -> "MockFlightSQLClient":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def get_tables(self, *args: Any, **kwargs: Any) -> dict:
        return {"assets": [{"name": "table1"}, {"name": "table2"}]}

    def get_table_info(self, table_name: str, *args: Any, **kwargs: Any) -> dict:
        if table_name == "table1":
            return {
                "path": "/public/table1",
                "fields": [
                    {"name": "id", "type": {"native_type": "INT", "nullable": False}},
                    {
                        "name": "name",
                        "type": {"native_type": "VARCHAR(255)", "nullable": True},
                    },
                    {"name": "age", "type": {"native_type": "INT", "nullable": True}},
                ],
                "extended_metadata": [
                    {"name": "primary_key", "value": {"key_columns": ["id"]}}
                ],
            }
        if table_name == "table2":
            return {
                "path": "/public/table2",
                "fields": [
                    {"name": "id", "type": {"native_type": "INT", "nullable": False}},
                    {
                        "name": "name",
                        "type": {"native_type": "VARCHAR(255)", "nullable": True},
                    },
                    {"name": "age", "type": {"native_type": "INT", "nullable": True}},
                ],
                "extended_metadata": [
                    {"name": "primary_key", "value": {"key_columns": ["id"]}}
                ],
            }
        error_msg = "Table not found"
        raise flight.FlightError(error_msg)

    def execute(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if "table1" in kwargs.get("query", ""):
            return pd.DataFrame({"id": [1], "name": ["test"], "age": [35]})

        if "table1" not in kwargs.get("query", ""):
            error_msg = "Table not found"
            raise flight.FlightError(error_msg)

        error_msg = "syntax error"
        raise ValueError(error_msg)

    def get_n_first_rows(
        self, schema: str, table_name: str, n: int = 3
    ) -> pd.DataFrame:
        return pd.DataFrame({"id": [1], "name": ["test"], "age": [35]})


### truncate_word


def test_truncate_word() -> None:
    assert truncate_word("This is a test", length=11) == "This is..."
    assert truncate_word("Short", length=10) == "Short"
    assert truncate_word("A", length=3) == "A"
    assert truncate_word("", length=10) == ""
    assert (
        truncate_word("This is a longer test", length=20, suffix="***")
        == "This is a longer***"
    )
    assert truncate_word(12345, length=10) == 12345  # Non-string input
    assert truncate_word("This is a test", length=0) == "This is a test"  # Length <= 0


def test_truncate_word_edge_cases() -> None:
    assert truncate_word("This is a test", length=8) == "This..."
    assert truncate_word("This is a test", length=11) == "This is..."
    assert truncate_word("This is a test", length=13) == "This is a..."
    assert (
        truncate_word("This is a test", length=16) == "This is a test"
    )  # Length >= string lengths


### pretty_print_table_info


def test_pretty_print_table_info(
    schema: str, table_name: str, table_info: dict
) -> None:
    expected_output = """
CREATE TABLE "test_schema"."test_table" (
\t"id" INT NOT NULL,
\t"name" VARCHAR(255),
\t"age" INT,
\tCONSTRAINT primary_key PRIMARY KEY (id)
\t)"""
    assert pretty_print_table_info(schema, table_name, table_info) == expected_output


def test_pretty_print_table_info_with_nullable_columns() -> None:
    schema = "another_schema"
    table_name = "another_table"
    table_info = {
        "fields": [
            {
                "name": "email",
                "type": {"native_type": "VARCHAR(255)", "nullable": True},
            },
            {
                "name": "created_at",
                "type": {"native_type": "TIMESTAMP", "nullable": True},
            },
        ],
        "extended_metadata": [
            {"name": "primary_key", "value": {"key_columns": ["email"]}}
        ],
    }
    expected_output = """
CREATE TABLE "another_schema"."another_table" (
\t"email" VARCHAR(255),
\t"created_at" TIMESTAMP,
\tCONSTRAINT primary_key PRIMARY KEY (email)
\t)"""
    assert pretty_print_table_info(schema, table_name, table_info) == expected_output


def test_pretty_print_table_info_without_primary_key() -> None:
    schema = "no_pk_schema"
    table_name = "no_pk_table"
    table_info = {
        "fields": [
            {"name": "value1", "type": {"native_type": "INT", "nullable": False}},
            {
                "name": "value2",
                "type": {"native_type": "VARCHAR(255)", "nullable": True},
            },
        ]
    }
    expected_output = """
CREATE TABLE "no_pk_schema"."no_pk_table" (
\t"value1" INT NOT NULL,
\t"value2" VARCHAR(255)
\t)"""
    assert pretty_print_table_info(schema, table_name, table_info) == expected_output


### WatsonSQLDatabase


def test_initialize_watsonx_sql_database_without_url(schema: str) -> None:
    with pytest.raises(ValueError) as e:
        WatsonxSQLDatabase(connection_id=CONNECTION_ID, schema=schema)

    assert "url" in str(e.value)
    assert "WATSONX_URL" in str(e.value)


def test_initialize_watsonx_sql_database_cloud_bad_path(schema: str) -> None:
    with pytest.raises(ValueError) as e:
        WatsonxSQLDatabase(
            connection_id=CONNECTION_ID,
            schema=schema,
            url="https://us-south.ml.cloud.ibm.com",
        )

    assert "apikey" in str(e.value) and "token" in str(e.value)
    assert "WATSONX_APIKEY" in str(e.value) and "WATSONX_TOKEN" in str(e.value)


def test_initialize_watsonx_sql_database_cpd_bad_path_without_all(schema: str) -> None:
    with pytest.raises(ValueError) as e:
        WatsonxSQLDatabase(
            connection_id=CONNECTION_ID,
            schema=schema,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
        )
    assert (
        "apikey" in str(e.value)
        and "password" in str(e.value)
        and "token" in str(e.value)
    )
    assert (
        "WATSONX_APIKEY" in str(e.value)
        and "WATSONX_PASSWORD" in str(e.value)
        and "WATSONX_TOKEN" in str(e.value)
    )


def test_initialize_watsonx_sql_database_cpd_bad_path_password_without_username(
    schema: str,
) -> None:
    with pytest.raises(ValueError) as e:
        WatsonxSQLDatabase(
            connection_id=CONNECTION_ID,
            schema=schema,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            password="test_password",
        )
    assert "username" in str(e.value)
    assert "WATSONX_USERNAME" in str(e.value)


def test_initialize_watsonx_sql_database_cpd_bad_path_apikey_without_username(
    schema: str,
) -> None:
    with pytest.raises(ValueError) as e:
        WatsonxSQLDatabase(
            connection_id=CONNECTION_ID,
            schema=schema,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            apikey="test_apikey",
        )

    assert "username" in str(e.value)
    assert "WATSONX_USERNAME" in str(e.value)


def test_initialize_watsonx_sql_database_cpd_bad_path_without_instance_id(
    schema: str,
) -> None:
    with pytest.raises(ValueError) as e:
        WatsonxSQLDatabase(
            connection_id=CONNECTION_ID,
            schema=schema,
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            apikey="test_apikey",
            username="test_user",
        )
    assert "instance_id" in str(e.value)
    assert "WATSONX_INSTANCE_ID" in str(e.value)


def test_initialize_watsonx_sql_database_without_any_params() -> None:
    with pytest.raises(TypeError):
        WatsonxSQLDatabase()  # type: ignore[call-arg]


def test_initialize_watsonx_sql_database_valid(
    schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_api_client = Mock()
    mock_api_client.default_project_id = PROJECT_ID

    with (
        mock.patch.dict(os.environ, clear=True),
        patch(
            "langchain_ibm.utilities.sql_database.APIClient",
            autospec=True,
            return_value=mock_api_client,
        ),
        patch(
            "langchain_ibm.utilities.sql_database.FlightSQLClient",
            autospec=True,
            return_value=MockFlightSQLClient(),
        ),
    ):
        envvars = {
            "WATSONX_APIKEY": "test_apikey",
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
        }
        for k, v in envvars.items():
            monkeypatch.setenv(k, v)

        wx_sql_database = WatsonxSQLDatabase(connection_id=CONNECTION_ID, schema=schema)

        assert isinstance(wx_sql_database._flight_sql_client, MockFlightSQLClient)
        assert wx_sql_database.schema == schema


def test_initialize_watsonx_sql_database_include_tables(
    schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_api_client = Mock()
    mock_api_client.default_project_id = PROJECT_ID

    with (
        mock.patch.dict(os.environ, clear=True),
        patch(
            "langchain_ibm.utilities.sql_database.APIClient",
            autospec=True,
            return_value=mock_api_client,
        ),
        patch(
            "langchain_ibm.utilities.sql_database.FlightSQLClient",
            autospec=True,
            return_value=MockFlightSQLClient(),
        ),
    ):
        envvars = {
            "WATSONX_APIKEY": "test_apikey",
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
        }
        for k, v in envvars.items():
            monkeypatch.setenv(k, v)

        wx_sql_database = WatsonxSQLDatabase(
            connection_id=CONNECTION_ID, schema=schema, include_tables=["table1"]
        )

        assert wx_sql_database.get_usable_table_names() == ["table1"]


def test_initialize_watsonx_sql_database_ignore_tables(
    schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_api_client = Mock()
    mock_api_client.default_project_id = PROJECT_ID

    with (
        mock.patch.dict(os.environ, clear=True),
        patch(
            "langchain_ibm.utilities.sql_database.APIClient",
            autospec=True,
            return_value=mock_api_client,
        ),
        patch(
            "langchain_ibm.utilities.sql_database.FlightSQLClient",
            autospec=True,
            return_value=MockFlightSQLClient(),
        ),
    ):
        envvars = {
            "WATSONX_APIKEY": "test_apikey",
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
        }
        for k, v in envvars.items():
            monkeypatch.setenv(k, v)

        wx_sql_database = WatsonxSQLDatabase(
            connection_id=CONNECTION_ID, schema=schema, ignore_tables=["table1"]
        )

        assert wx_sql_database.get_usable_table_names() == ["table2"]


def test_initialize_watsonx_sql_database_get_table_info(
    schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_api_client = Mock()
    mock_api_client.default_project_id = PROJECT_ID

    with (
        mock.patch.dict(os.environ, clear=True),
        patch(
            "langchain_ibm.utilities.sql_database.APIClient",
            autospec=True,
            return_value=mock_api_client,
        ),
        patch(
            "langchain_ibm.utilities.sql_database.FlightSQLClient",
            autospec=True,
            return_value=MockFlightSQLClient(),
        ),
    ):
        envvars = {
            "WATSONX_APIKEY": "test_apikey",
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
        }
        for k, v in envvars.items():
            monkeypatch.setenv(k, v)

        wx_sql_database = WatsonxSQLDatabase(connection_id=CONNECTION_ID, schema=schema)
        expected_output = """
CREATE TABLE "test_schema"."table1" (
\t"id" INT NOT NULL,
\t"name" VARCHAR(255),
\t"age" INT,
\tCONSTRAINT primary_key PRIMARY KEY (id)
\t)

First 3 rows of table table1:

 id name  age
  1 test   35"""
        print(wx_sql_database.get_table_info(["table1"]))
        assert wx_sql_database.get_table_info(["table1"]) == expected_output


def test_initialize_watsonx_sql_database_get_table_info_no_throw(
    schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_api_client = Mock()
    mock_api_client.default_project_id = PROJECT_ID

    with (
        mock.patch.dict(os.environ, clear=True),
        patch(
            "langchain_ibm.utilities.sql_database.APIClient",
            autospec=True,
            return_value=mock_api_client,
        ),
        patch(
            "langchain_ibm.utilities.sql_database.FlightSQLClient",
            autospec=True,
            return_value=MockFlightSQLClient(),
        ),
    ):
        envvars = {
            "WATSONX_APIKEY": "test_apikey",
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
        }
        for k, v in envvars.items():
            monkeypatch.setenv(k, v)

        wx_sql_database = WatsonxSQLDatabase(connection_id=CONNECTION_ID, schema=schema)
        with pytest.raises(ValueError):
            wx_sql_database.get_table_info(["tableX"])
        assert "tableX" in wx_sql_database.get_table_info_no_throw(["tableX"])


def test_initialize_watsonx_sql_database_run(
    schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_api_client = Mock()
    mock_api_client.default_project_id = PROJECT_ID

    with (
        mock.patch.dict(os.environ, clear=True),
        patch(
            "langchain_ibm.utilities.sql_database.APIClient",
            autospec=True,
            return_value=mock_api_client,
        ),
        patch(
            "langchain_ibm.utilities.sql_database.FlightSQLClient",
            autospec=True,
            return_value=MockFlightSQLClient(),
        ),
    ):
        envvars = {
            "WATSONX_APIKEY": "test_apikey",
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
        }
        for k, v in envvars.items():
            monkeypatch.setenv(k, v)

        wx_sql_database = WatsonxSQLDatabase(connection_id=CONNECTION_ID, schema=schema)

        assert (
            wx_sql_database.run(f"SELECT * FROM {schema}.table1", include_columns=True)
            == "[{'id': 1, 'name': 'test', 'age': 35}]"
        )


def test_initialize_watsonx_sql_database_run_no_throw(
    schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_api_client = Mock()
    mock_api_client.default_project_id = PROJECT_ID

    with (
        mock.patch.dict(os.environ, clear=True),
        patch(
            "langchain_ibm.utilities.sql_database.APIClient",
            autospec=True,
            return_value=mock_api_client,
        ),
        patch(
            "langchain_ibm.utilities.sql_database.FlightSQLClient",
            autospec=True,
            return_value=MockFlightSQLClient(),
        ),
    ):
        envvars = {
            "WATSONX_APIKEY": "test_apikey",
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
        }
        for k, v in envvars.items():
            monkeypatch.setenv(k, v)

        wx_sql_database = WatsonxSQLDatabase(connection_id=CONNECTION_ID, schema=schema)

        assert "Table not found" in wx_sql_database.run_no_throw(
            f"SELECT * FROM {schema}.tableX", include_columns=True
        )
