from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseLanguageModel

from langchain_ibm.agent_toolkits.sql.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
)
from langchain_ibm.agent_toolkits.sql.toolkit import WatsonxSQLDatabaseToolkit
from langchain_ibm.utilities.sql_database import WatsonxSQLDatabase


@pytest.fixture
def mock_db() -> Mock:
    db = Mock(spec=WatsonxSQLDatabase)
    db.schema = "test_schema"
    return db


@pytest.fixture
def mock_llm() -> Mock:
    return Mock(spec=BaseLanguageModel)


def test_watsonx_sql_database_toolkit_initialization(
    mock_db: Mock, mock_llm: Mock
) -> None:
    """Test initializing WatsonxSQLDatabaseToolkit."""
    toolkit = WatsonxSQLDatabaseToolkit(db=mock_db, llm=mock_llm)
    assert toolkit.db == mock_db
    assert toolkit.llm == mock_llm


def test_watsonx_sql_database_toolkit_get_tools(
    mock_db: Mock, mock_llm: Mock
) -> None:
    """Test get_tools returns all four expected tools with correct configuration."""
    toolkit = WatsonxSQLDatabaseToolkit(db=mock_db, llm=mock_llm)
    tools = toolkit.get_tools()

    assert len(tools) == 4
    query_tool, info_tool, list_tool, checker_tool = tools

    assert isinstance(query_tool, QuerySQLDatabaseTool)
    assert query_tool.db == mock_db

    assert isinstance(info_tool, InfoSQLDatabaseTool)
    assert info_tool.db == mock_db

    assert isinstance(list_tool, ListSQLDatabaseTool)
    assert list_tool.db == mock_db

    assert isinstance(checker_tool, QuerySQLCheckerTool)
    assert checker_tool.db == mock_db
    assert checker_tool.llm == mock_llm

    assert "sql_db_list_tables" in info_tool.description
    assert "sql_db_schema" in query_tool.description
    assert "sql_db_query" in checker_tool.description


def test_watsonx_sql_database_toolkit_get_context(
    mock_db: Mock, mock_llm: Mock
) -> None:
    """Test get_context delegates to db.get_context."""
    expected_context = {"table_info": "info", "table_names": "table1, table2"}
    mock_db.get_context.return_value = expected_context

    toolkit = WatsonxSQLDatabaseToolkit(db=mock_db, llm=mock_llm)
    context = toolkit.get_context()

    assert context == expected_context
    mock_db.get_context.assert_called_once()
