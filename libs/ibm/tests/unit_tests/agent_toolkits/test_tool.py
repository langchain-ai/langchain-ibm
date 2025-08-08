from typing import Any
from unittest.mock import Mock

import pytest
from pyarrow import flight  # type: ignore[import-untyped]

from langchain_ibm.agent_toolkits.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDatabaseTool,
)
from langchain_ibm.utilities.sql_database import WatsonxSQLDatabase


@pytest.fixture
def mock_db() -> Mock:
    return Mock(spec=WatsonxSQLDatabase)


### QuerySQLDatabaseTool


def test_query_tool_run_with_valid_query(mock_db: Mock) -> None:
    """Test running a valid query."""
    mock_db.run_no_throw.return_value = "Valid query result"
    tool = QuerySQLDatabaseTool(db=mock_db)

    query = "SELECT * FROM table"
    result = tool._run(query, None)
    assert result == "Valid query result"
    mock_db.run_no_throw.assert_called_once_with(query)


def test_query_tool_run_with_invalid_query(mock_db: Mock) -> None:
    """Test running an invalid query."""

    def mock_run_no_throw(*args: Any, **kwargs: Any) -> None:
        raise flight.FlightError("Invalid query")

    mock_db.run.side_effect = mock_run_no_throw
    tool = QuerySQLDatabaseTool(db=mock_db)

    query = "SELECT * FROM non_existent_table"
    result = tool._run(query, None)
    assert result.startswith("Error"), "Expected an error message"
    mock_db.run_no_throw.assert_called_once_with(query)


### InfoSQLDatabaseTool


def test_info_tool_run_with_valid_query(mock_db: Mock) -> None:
    """Test running a valid query."""
    mock_db.get_table_info_no_throw.return_value = "schema_info"
    tool = InfoSQLDatabaseTool(db=mock_db)

    result = tool._run("table1,table2", None)

    mock_db.get_table_info_no_throw.assert_called_once_with(["table1", "table2"])
    assert result == "schema_info"


def test_info_tool_run_with_invalid_query(mock_db: Mock) -> None:
    """Test running an invalid query."""

    def mock_get_table_info(*args: Any, **kwargs: Any) -> None:
        raise flight.FlightError("Table not Found")

    mock_db.get_table_info.side_effect = mock_get_table_info
    tool = InfoSQLDatabaseTool(db=mock_db)

    result = tool._run("tableX", None)

    mock_db.get_table_info_no_throw.assert_called_once_with(["tableX"])
    assert result.startswith("Error"), "Expected an error message"


### ListSQLDatabaseTool


def test_list_tool_run_with_valid_query(mock_db: Mock) -> None:
    """Test running a valid query."""
    mock_db.get_usable_table_names.return_value = ["table1", "table2"]
    tool = ListSQLDatabaseTool(db=mock_db)

    result = tool._run()

    mock_db.get_usable_table_names.assert_called_once_with()
    assert result == "table1, table2"
