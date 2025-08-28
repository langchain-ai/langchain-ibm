"""Tools for interacting with a watsonx SQL databases via pyarrow.flight.FlightClient.

Based on the langchain_community.tools.sql_database.tool module."""

from typing import Any, Dict, Optional, Type, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from langchain_ibm.utilities.sql_database import WatsonxSQLDatabase

QUERY_CHECKER = """
{query}
Double check the query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- Make sure that schema name `{schema}` is added to the table name, e.g. {schema}.table1

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only.

SQL Query: """  # noqa: E501


class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    db: WatsonxSQLDatabase = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _QuerySQLDatabaseToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct SQL query.")


class QuerySQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for querying a SQL database."""

    name: str = "sql_db_query"
    description: str = """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query correctness, 
    and try again.
    """
    args_schema: Type[BaseModel] = _QuerySQLDatabaseToolInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)


class _InfoSQLDatabaseToolInput(BaseModel):
    table_names: str = Field(
        ...,
        description=(
            "A comma-separated list of the table names "
            "for which to return the schema. "
            "Example input: 'table1, table2, table3'"
        ),
    )


class InfoSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting metadata about a SQL database."""

    name: str = "sql_db_schema"
    description: str = "Get the schema and sample rows for the specified SQL tables."
    args_schema: Type[BaseModel] = _InfoSQLDatabaseToolInput

    def _run(
        self,
        table_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.db.get_table_info_no_throw(
            [t.strip() for t in table_names.split(",")]
        )


class _ListSQLDatabaseToolInput(BaseModel):
    tool_input: str = Field("", description="An empty string")


class ListSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting tables names."""

    name: str = "sql_db_list_tables"
    description: str = (
        "Input is an empty string, output is a comma-separated list "
        "of tables in the database."
    )
    args_schema: Type[BaseModel] = _ListSQLDatabaseToolInput

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get a comma-separated list of table names."""
        return ", ".join(self.db.get_usable_table_names())


class _QuerySQLCheckerToolInput(BaseModel):
    query: str = Field(..., description="A detailed and SQL query to be checked.")


class QuerySQLCheckerTool(BaseSQLDatabaseTool, BaseTool):
    """Use an LLM to check if a query is correct."""

    template: str = QUERY_CHECKER
    llm: BaseLanguageModel
    llm_chain: Any = Field(init=False)
    name: str = "sql_db_query_checker"
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with sql_db_query!
    """
    args_schema: Type[BaseModel] = _QuerySQLCheckerToolInput

    @model_validator(mode="before")
    @classmethod
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Any:
        if "llm_chain" not in values:
            prompt = PromptTemplate(
                template=QUERY_CHECKER, input_variables=["query", "schema"]
            )
            llm = cast(BaseLanguageModel, values.get("llm"))

            values["llm_chain"] = prompt | llm

        if values["llm_chain"].first.input_variables != ["query", "schema"]:
            raise ValueError(
                "LLM chain for QueryCheckerTool must have input variables ['query', 'schema']"  # noqa: E501
            )

        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the LLM to check the query."""
        resp = self.llm_chain.invoke(
            {"query": query, "schema": self.db.schema},
            callbacks=run_manager.get_child() if run_manager else None,
        )
        if isinstance(resp, BaseMessage):
            return str(resp.content)
        return resp

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        resp = await self.llm_chain.ainvoke(
            {"query": query, "schema": self.db.schema},
            callbacks=run_manager.get_child() if run_manager else None,
        )
        if isinstance(resp, BaseMessage):
            return str(resp.content)
        return resp
