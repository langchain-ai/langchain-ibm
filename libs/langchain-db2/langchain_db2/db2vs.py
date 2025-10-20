from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import ibm_db_dbi  # type: ignore

if TYPE_CHECKING:
    from ibm_db_dbi import Connection

import numpy as np
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_db2.utils import EmbeddingsSchema

logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Define a type variable that can be any kind of function
T = TypeVar("T", bound=Callable[..., Any])


def _handle_exceptions(func: T) -> T:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except RuntimeError as db_err:
            # Handle a known type of error (e.g., DB-related) specifically
            logger.exception("DB-related error occurred.")
            raise RuntimeError(
                "Failed due to a DB issue: {}".format(db_err)
            ) from db_err
        except ValueError as val_err:
            # Handle another known type of error specifically
            logger.exception("Validation error.")
            raise ValueError("Validation failed: {}".format(val_err)) from val_err
        except Exception as e:
            # Generic handler for all other exceptions
            logger.exception("An unexpected error occurred: {}".format(e))
            raise RuntimeError("Unexpected error: {}".format(e)) from e

    return cast(T, wrapper)

def _get_distance_function(distance_strategy: DistanceStrategy) -> str:
    # Dictionary to map distance strategies to their corresponding function
    # names
    distance_strategy2function = {
        DistanceStrategy.EUCLIDEAN_DISTANCE: "EUCLIDEAN",
        DistanceStrategy.DOT_PRODUCT: "DOT",
        DistanceStrategy.COSINE: "COSINE",
    }

    # Attempt to return the corresponding distance function
    if distance_strategy in distance_strategy2function:
        return distance_strategy2function[distance_strategy]

    # If it's an unsupported distance strategy, raise an error
    raise ValueError(f"Unsupported distance strategy: {distance_strategy}")

def _object_exists(client: Connection, query: str, params: tuple = ()) -> bool:
    """Utility to check if a catalog query returns at least one row."""
    cur = client.cursor()
    try:
        cur.execute(query, params)
        row = cur.fetchone()
        return row is not None
    finally:
        cur.close()

def _bufferpool_exists(client: Connection, bp_name: str) -> bool:
    # SYSCAT.BUFFERPOOLS: bpname is uppercased in catalog
    q = "SELECT 1 FROM SYSCAT.BUFFERPOOLS WHERE UPPER(BPNAME)=UPPER(?)"
    return _object_exists(client, q, (bp_name,))

def _tablespace_exists(client: Connection, ts_name: str) -> bool:
    # SYSCAT.TABLESPACES: tbspace is uppercased in catalog
    q = "SELECT 1 FROM SYSCAT.TABLESPACES WHERE UPPER(TBSPACE)=UPPER(?)"
    return _object_exists(client, q, (ts_name,))

def _table_exists(client: Connection, table_name: str) -> bool:
    # Handle optional schema qualification; default current schema.
    # Split into schema and name; else rely on CURRENT SCHEMA in the session.
    parts = table_name.split(".")
    if len(parts) == 2:
        schema, name = parts[0], parts[1]
        q = """
            SELECT 1
            FROM SYSCAT.TABLES
            WHERE UPPER(TABSCHEMA)=UPPER(?) AND UPPER(TABNAME)=UPPER(?)
        """
        return _object_exists(client, q, (schema, name))
    else:
        q = """
            SELECT 1
            FROM SYSCAT.TABLES
            WHERE UPPER(TABSCHEMA)=UPPER(CURRENT SCHEMA) AND UPPER(TABNAME)=UPPER(?)
        """
        return _object_exists(client, q, (table_name,))

def _ensure_32k_bufferpool(client: Connection, bp_name: str = "BP32K", size_pages: int = 1000):
    """
    Ensure a 32K bufferpool exists. `size_pages` is in bufferpool pages, not bytes.
    """
    if _bufferpool_exists(client, bp_name):
        logger.info(f"Bufferpool {bp_name} already exists.")
        return

    ddl = f"CREATE BUFFERPOOL {bp_name} SIZE {size_pages} PAGESIZE 32K"
    cur = client.cursor()
    try:
        logger.info(f"Creating bufferpool {bp_name} (32K, size {size_pages} pages)...")
        cur.execute(ddl)
        cur.execute("COMMIT")
        logger.info(f"Bufferpool {bp_name} created.")
    finally:
        cur.close()

def _ensure_32k_tablespace(
    client: Connection,
    ts_name: str = "TS32K",
    bp_name: str = "BP32K",
    stogroup: str = "IBMSTOGROUP",
    extent_kpages: int = 32,
):
    """
    Ensure a 32K tablespace exists on automatic storage bound to the given bufferpool.
    extent_kpages: EXTENTSIZE in pages (32K pages). 32 is a reasonable default.
    """
    if _tablespace_exists(client, ts_name):
        logger.info(f"Tablespace {ts_name} already exists.")
        return

    # Ensure bufferpool exists before creating the tablespace
    _ensure_32k_bufferpool(client, bp_name)

    # Automatic storage + explicit bufferpool binding
    ddl = (
        f"CREATE LARGE TABLESPACE {ts_name} "
        f"PAGESIZE 32K "
        f"MANAGED BY AUTOMATIC STORAGE "
        f"USING STOGROUP {stogroup} "
        f"EXTENTSIZE {extent_kpages} "
        f"BUFFERPOOL {bp_name}"
    )
    cur = client.cursor()
    try:
        logger.info(f"Creating tablespace {ts_name} (32K, BP={bp_name}, STG={stogroup})...")
        cur.execute(ddl)
        cur.execute("COMMIT")
        logger.info(f"Tablespace {ts_name} created.")
    finally:
        cur.close()

@_handle_exceptions
def _create_table(
    client: Connection,
    table_name: str,
    embedding_dim: int,
    text_field:
    str = "text",
    ts_name: str = "TS32K",
    bp_name: str = "BP32K",
    stogroup: str = "IBMSTOGROUP"
) -> None:
    """
    Create (if needed) a 32K bufferpool + tablespace, then create the table IN that tablespace.
    """

    cols_dict = {
        "id": "CHAR(16) PRIMARY KEY NOT NULL",
        text_field: "CLOB",
        "metadata": "BLOB",
        "embedding": f"vector({embedding_dim}, FLOAT32)",
    }

    # Ensure infra exists
    _ensure_32k_tablespace(client, ts_name=ts_name, bp_name=bp_name, stogroup=stogroup)

    if not _table_exists(client, table_name):
        cursor = client.cursor()
        ddl_body = ", ".join(
            f"{col_name} {col_type}" for col_name, col_type in cols_dict.items()
        )
        ddl = f"CREATE TABLE {table_name} ({ddl_body})"
        try:
            cursor.execute(ddl)
            cursor.execute("COMMIT")
            logger.info(f"Table {table_name} created successfully...")
        finally:
            cursor.close()
    else:
        logger.info(f"Table {table_name} already exists...")


@_handle_exceptions
def drop_table(client: Connection, table_name: str) -> None:
    """Drop a table from the database.

    Args:
        client: The `ibm_db_dbi` connection object
        table_name: The name of the table to drop

    Raises:
        RuntimeError: If an error occurs while dropping the table

    ??? example "Example"

        ```python
        from langchain_db2.db2vs import drop_table

        drop_table(
            client=db_client,  # ibm_db_dbi.Connection
            table_name="TABLE_NAME",
        )
        ```

    """
    if _table_exists(client, table_name):
        cursor = client.cursor()
        ddl = f"DROP TABLE {table_name}"
        try:
            cursor.execute(ddl)
            cursor.execute("COMMIT")
            logger.info(f"Table {table_name} dropped successfully...")
        finally:
            cursor.close()
    else:
        logger.info(f"Table {table_name} not found...")
    return


@_handle_exceptions
def clear_table(client: Connection, table_name: str) -> None:
    """Remove all records from the table using TRUNCATE.

    Args:
        client: The ibm_db_dbi connection object
        table_name: The name of the table to clear

    ??? example "Example"

        ```python
        from langchain_db2.db2vs import clear_table

        clear_table(
            client=db_client,  # ibm_db_dbi.Connection
            table_name="TABLE_NAME",
        )
        ```
    """
    if not _table_exists(client, table_name):
        logger.info(f"Table {table_name} not found…")
        return

    cursor = client.cursor()
    ddl = f"TRUNCATE TABLE {table_name} IMMEDIATE"
    try:
        client.commit()
        cursor.execute(ddl)
        client.commit()
        logger.info(f"Table {table_name} cleared successfully.")
    except Exception:
        client.rollback()
        logger.exception(f"Failed to clear table {table_name}. Rolled back.")
        raise
    finally:
        cursor.close()


class DB2VS(VectorStore):
    """`DB2VS` vector store.

    Args:
        embedding_function: The embedding backend used to generate vectors for stored
            texts and queries
        table_name: DB2 table name
        client: Existing DB2 connection. Required if `connection_args` is not provided
        distance_strategy: Similarity metric used by Db2 `VECTOR_DISTANCE` when
            ranking results
        query: Probe text used once to infer embedding dimension
        params: Extra options
        connection_args: Connection parameters used when `client` is not supplied.
            Expected keys: `{"database": str, "host": str, "port": str,
            "username": str, "password": str, "security": bool}`
        text_field: Column name for the raw text (CLOB)

    ???+ info "Setup"

        To use, you should have:

        - the `langchain_db2` python package installed
        - a connection to db2 database with vector store feature (v12.1.2+)

        ```bash
        pip install -U langchain-db2

        # or using uv
        uv add langchain-db2
        ```

    ??? info "Instantiate"

        Create a Vector Store instance with `ibm_db_dbi.Connection` object

        ```python
        from langchain_db2 import DB2VS

        db2vs = DB2VS(
            embedding_function=embeddings, table_name=table_name, client=db_client
        )
        ```

        Create a Vector Store instance with `connection_args`

        ```python
        from langchain_db2 import DB2VS

        db2vs = DB2VS(
            embedding_function=embeddings,
            table_name=table_name,
            connection_args={
                "database": "<DATABASE>",
                "host": "<HOST>",
                "port": "<PORT>",
                "username": "<USERNAME>",
                "password": "<PASSWORD>",
                "security": False,
            },
        )
        ```

    """

    def __init__(
        self,
        embedding_function: Union[
            Callable[[str], List[float]],
            Embeddings,
        ],
        table_name: str,
        client: Optional[Connection] = None,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        query: Optional[str] = "What is a Db2 database",
        params: Optional[Dict[str, Any]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        text_field: str = "text",
    ):
        if client is None:
            if connection_args is not None:
                database = connection_args.get("database")
                host = connection_args.get("host")
                port = connection_args.get("port")
                username = connection_args.get("username")
                password = connection_args.get("password")

                conn_str = (
                    f"DATABASE={database};hostname={host};port={port};"
                    f"uid={username};pwd={password};"
                )

                if "security" in connection_args:
                    security = connection_args.get("security")
                    conn_str += f"security={security};"

                self.client = ibm_db_dbi.connect(conn_str, "", "")
            else:
                raise ValueError("No valid connection or connection_args is passed")
        else:
            """Initialize with ibm_db_dbi client."""
            self.client = client
        try:
            """Initialize with necessary components."""
            if not isinstance(embedding_function, EmbeddingsSchema):
                logger.warning(
                    "`embedding_function` is expected to be an Embeddings "
                    "object, support for passing in a function will soon "
                    "be removed."
                )
            self.embedding_function = embedding_function
            self.query = query
            embedding_dim = self.get_embedding_dimension()

            self.table_name = table_name
            self.distance_strategy = distance_strategy
            self.params = params
            self._text_field = text_field
            _create_table(
                self.client, self.table_name, embedding_dim, text_field=self._text_field
            )
        except ibm_db_dbi.DatabaseError as db_err:
            logger.exception(f"Database error occurred while create table: {db_err}")
            raise RuntimeError(
                "Failed to create table due to a database error."
            ) from db_err
        except ValueError as val_err:
            logger.exception(f"Validation error: {val_err}")
            raise RuntimeError(
                "Failed to create table due to a validation error."
            ) from val_err
        except Exception as ex:
            logger.exception("An unexpected error occurred while creating the table.")
            raise RuntimeError(
                "Failed to create table due to an unexpected error."
            ) from ex

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """
        A property that returns an Embeddings instance if embedding_function
        is an instance of Embeddings, otherwise returns None.

        Returns:
            Embeddings instance if embedding_function is an instance of
                Embeddings, otherwise returns None
        """
        return (
            self.embedding_function
            if isinstance(self.embedding_function, EmbeddingsSchema)
            else None
        )

    def get_embedding_dimension(self) -> int:
        # Embed the single document by wrapping it in a list
        embedded_document = self._embed_documents(
            [self.query if self.query is not None else ""]
        )

        # Get the first (and only) embedding's dimension
        return len(embedded_document[0])

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        if isinstance(self.embedding_function, EmbeddingsSchema):
            return self.embedding_function.embed_documents(texts)
        elif callable(self.embedding_function):
            return [self.embedding_function(text) for text in texts]
        else:
            raise TypeError(
                "The embedding_function is neither Embeddings nor callable."
            )

    def _embed_query(self, text: str) -> List[float]:
        if isinstance(self.embedding_function, EmbeddingsSchema):
            return self.embedding_function.embed_query(text)
        else:
            return self.embedding_function(text)

    @_handle_exceptions
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore
            metadatas: Optional list of metadatas associated with the texts
            ids: Optional list of ids for the texts that are being added to
                the vector store
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore
        """

        texts = list(texts)

        if metadatas and len(metadatas) != len(texts):
            msg = (
                f"metadatas must be the same length as texts. "
                f"Got {len(metadatas)} metadatas and {len(texts)} texts."
            )
            raise ValueError(msg)

        if ids:
            if len(ids) != len(texts):
                msg = (
                    f"ids must be the same length as texts. "
                    f"Got {len(ids)} ids and {len(texts)} texts."
                )
                raise ValueError(msg)
            # If ids are provided, hash them to maintain consistency
            processed_ids = [
                hashlib.sha256(_id.encode()).hexdigest()[:16].upper() for _id in ids
            ]
        elif metadatas:
            if all("id" in metadata for metadata in metadatas):
                # If no ids are provided but metadatas with ids are, generate
                # ids from metadatas
                processed_ids = [
                    hashlib.sha256(metadata["id"].encode()).hexdigest()[:16].upper()
                    for metadata in metadatas
                ]
            else:
                # In the case partial metadata has id, generate new id if metadate
                # doesn't have it.
                processed_ids = []
                for metadata in metadatas:
                    if "id" in metadata:
                        processed_ids.append(
                            hashlib.sha256(metadata["id"].encode())
                            .hexdigest()[:16]
                            .upper()
                        )
                    else:
                        processed_ids.append(
                            hashlib.sha256(str(uuid.uuid4()).encode())
                            .hexdigest()[:16]
                            .upper()
                        )
        else:
            # Generate new ids if none are provided
            generated_ids = [
                str(uuid.uuid4()) for _ in texts
            ]  # uuid4 is more standard for random UUIDs
            processed_ids = [
                hashlib.sha256(_id.encode()).hexdigest()[:16].upper()
                for _id in generated_ids
            ]

        embeddings = self._embed_documents(texts)
        if not metadatas:
            metadatas = [{} for _ in texts]

        embedding_len = self.get_embedding_dimension()
        docs: List[Tuple[Any, Any, Any, Any]]
        docs = [
            (id_, f"{embedding}", json.dumps(metadata), text)
            for id_, embedding, metadata, text in zip(
                processed_ids, embeddings, metadatas, texts
            )
        ]

        SQL_INSERT = (
            f"INSERT INTO "
            f"{self.table_name} (id, embedding, metadata, {self._text_field}) "
            f"VALUES (?, VECTOR(?, {embedding_len}, FLOAT32), SYSTOOLS.JSON2BSON(?), ?)"
        )

        cursor = self.client.cursor()
        try:
            cursor.executemany(SQL_INSERT, docs)
            cursor.execute("COMMIT")
        finally:
            cursor.close()
        return processed_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: The natural-language text to search for
            k: Number of Documents to return
            filter: Filter by metadata

        Returns:
            Documents most similar to a query
        """
        if isinstance(self.embedding_function, EmbeddingsSchema):
            embedding = self.embedding_function.embed_query(query)
        documents = self.similarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return documents

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to a query embedding.

        Args:
            embedding: Embedding to look up documents similar to
            k: Number of Documents to return
            filter: Filter by metadata

        Returns:
            Documents ordered from most to least similar
        """
        docs_and_scores = self.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return the top-k documents most similar to a text query, with scores.

        Args:
            query: Natural-language query to embed and search with
            k: Number of results to return
            filter: Filter by metadata

        Returns:
            A list of (document, score) pairs ordered by similarity.
                The score is the vector **distance**; lower values indicate
                closer matches.
        """
        if isinstance(self.embedding_function, EmbeddingsSchema):
            embedding = self.embedding_function.embed_query(query)
        docs_and_scores = self.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return docs_and_scores

    @_handle_exceptions
    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return top-k documents for a query embedding, with relevance scores.

        Args:
            embedding: Embedding to look up documents similar to
            k: Number of Documents to return
            filter: Filter by metadata

        Returns:
            A list of `(Document, distance)` pairs ordered from most to least
                similar (smallest distance first).
        """
        docs_and_scores = []
        embedding_len = self.get_embedding_dimension()

        # If a vector index exists on the embedding with a matching distance type,
        # approximate nearest neighbor (ANN) search will be used by default.
        query = f"""
        SELECT id,
          {self._text_field},
          SYSTOOLS.BSON2JSON(metadata),
          vector_distance(embedding, VECTOR('{embedding}', {embedding_len}, FLOAT32),
          {_get_distance_function(self.distance_strategy)}) as distance
        FROM {self.table_name}
        ORDER BY distance
        FETCH FIRST {k} ROWS ONLY
        """

        # Execute the query
        cursor = self.client.cursor()
        try:
            cursor.execute(query)
            results = cursor.fetchall()

            # Filter results if filter is provided
            for result in results:
                metadata = json.loads(result[2] if result[2] is not None else "{}")

                # Apply filtering based on the 'filter' dictionary
                if filter:
                    if all(metadata.get(key) in value for key, value in filter.items()):
                        doc = Document(
                            page_content=(result[1] if result[1] is not None else ""),
                            metadata=metadata,
                        )
                        distance = result[3]
                        docs_and_scores.append((doc, distance))
                else:
                    doc = Document(
                        page_content=(result[1] if result[1] is not None else ""),
                        metadata=metadata,
                    )
                    distance = result[3]
                    docs_and_scores.append((doc, distance))
        finally:
            cursor.close()
        return docs_and_scores

    @_handle_exceptions
    def similarity_search_by_vector_returning_embeddings(
        self,
        embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float, np.ndarray]]:
        """Return top-k documents, their distances, and stored embeddings.

        Args:
            embedding: Embedding to look up documents similar to
            k: Number of Documents to return
            filter: Filter by metadata

        Returns:
            Tuples of `(document, distance, embedding_array)`, ordered from
                most to least similar (ascending distance)
        """
        documents = []
        embedding_len = self.get_embedding_dimension()

        # If a vector index exists on the embedding with a matching distance type,
        # approximate nearest neighbor (ANN) search will be used by default.
        query = f"""
        SELECT id,
          {self._text_field},
          SYSTOOLS.BSON2JSON(metadata),
          vector_distance(embedding, VECTOR('{embedding}', {embedding_len}, FLOAT32),
          {_get_distance_function(self.distance_strategy)}) as distance,
          embedding
        FROM {self.table_name}
        ORDER BY distance
        FETCH FIRST {k} ROWS ONLY
        """

        # Execute the query
        cursor = self.client.cursor()
        try:
            cursor.execute(query)
            results = cursor.fetchall()

            for result in results:
                page_content_str = result[1] if result[1] is not None else ""
                metadata = json.loads(result[2] if result[2] is not None else "{}")

                # Apply filter if provided and matches; otherwise, add all
                # documents
                if not filter or all(
                    metadata.get(key) in value for key, value in filter.items()
                ):
                    document = Document(
                        page_content=page_content_str, metadata=metadata
                    )
                    distance = result[3]

                    # Assuming result[4] is already in the correct format;
                    # adjust if necessary
                    current_embedding = (
                        np.array(json.loads(result[4]), dtype=np.float32)
                        if result[4]
                        else np.empty(0, dtype=np.float32)
                    )

                    documents.append((document, distance, current_embedding))
        finally:
            cursor.close()
        return documents  # type: ignore

    @_handle_exceptions
    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores selected using the
        maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity among selected documents.

        Args:
            embedding: Embedding to look up documents similar to
            k: Number of Documents to return
            fetch_k: Number of Documents to fetch before filtering to
                pass to MMR algorithm
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity
            filter: Filter by metadata

        Returns:
            List of Documents and similarity scores selected by maximal
                marginal relevance and score for each.
        """

        # Fetch documents and their scores
        docs_scores_embeddings = self.similarity_search_by_vector_returning_embeddings(
            embedding, fetch_k, filter=filter
        )
        # Assuming documents_with_scores is a list of tuples (Document, score)

        # If you need to split documents and scores for processing (e.g.,
        # for MMR calculation)
        documents, scores, embeddings = (
            zip(*docs_scores_embeddings) if docs_scores_embeddings else ([], [], [])
        )

        # Assume maximal_marginal_relevance method accepts embeddings and
        # scores, and returns indices of selected docs
        mmr_selected_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            list(embeddings),
            k=k,
            lambda_mult=lambda_mult,
        )

        # Filter documents based on MMR-selected indices and map scores
        mmr_selected_documents_with_scores = [
            (documents[i], scores[i]) for i in mmr_selected_indices
        ]

        return mmr_selected_documents_with_scores

    @_handle_exceptions
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity among selected documents.

        Args:
            embedding: Embedding to look up documents similar to
            k: Number of Documents to return
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity
            filter: Filter by metadata

        Returns:
            List of Documents selected by maximal marginal relevance
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    @_handle_exceptions
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity among selected documents.

        Args:
            query: Text to look up documents similar to
            k: Number of Documents to return
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity
            filter: Filter by metadata

        Returns:
            List of Documents selected by maximal marginal relevance

        `max_marginal_relevance_search` requires that `query` returns matched
        embeddings alongside the match documents.
        """
        embedding = self._embed_query(query)
        documents = self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return documents

    @_handle_exceptions
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs

        Args:
            ids: List of ids to delete
        """
        if ids is None:
            raise ValueError("No ids provided to delete.")

        is_hashed = bool(ids) and all(re.fullmatch(r"[A-F0-9]{16}", _id) for _id in ids)

        if is_hashed:
            hashed_ids = ids  # use as-is
        else:
            # Compute SHA-256 hashes of the raw ids and truncate them
            hashed_ids = [
                hashlib.sha256(_id.encode("utf-8")).hexdigest()[:16].upper()
                for _id in ids
            ]

        # Constructing the SQL statement with individual placeholders
        placeholders = ", ".join("?" for _ in hashed_ids)

        ddl = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"
        cursor = self.client.cursor()
        try:
            cursor.execute(ddl, hashed_ids)
            cursor.execute("COMMIT")
        finally:
            cursor.close()

    @classmethod
    @_handle_exceptions
    def from_texts(
        cls: Type[DB2VS],
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> DB2VS:
        """Return VectorStore initialized from texts and embeddings.

        Args:
            texts: Iterable of strings to add to the vectorstore
            embedding: Embedding to look up documents similar to
            metadatas: Optional list of metadatas associated with the texts

        Returns:
            A ready-to-use vector store with the provided texts loaded
        """
        client = kwargs.get("client")
        if client is None:
            raise ValueError("client parameter is required...")
        params = kwargs.get("params", {})

        table_name = str(kwargs.get("table_name", "langchain"))

        distance_strategy = cast(
            DistanceStrategy, kwargs.get("distance_strategy", None)
        )
        if not isinstance(distance_strategy, DistanceStrategy):
            raise TypeError(
                f"Expected DistanceStrategy got {type(distance_strategy).__name__} "
            )

        query = kwargs.get("query", "What is a Db2 database")

        drop_table(client, table_name)

        vss = cls(
            client=client,
            embedding_function=embedding,
            table_name=table_name,
            distance_strategy=distance_strategy,
            query=query,
            params=params,
        )
        vss.add_texts(texts=list(texts), metadatas=metadatas)
        return vss

    @_handle_exceptions
    def get_pks(self, expr: Optional[str] = None) -> List[str]:
        """Get primary keys, optionally filtered by expr.

        Args:
            expr: SQL boolean expression to filter rows, e.g.:
                `id IN ('ABC123','DEF456')` or `title LIKE 'Abc%'`.
                If None, returns all rows.

        Returns:
            List of matching primary-key values.
        """
        sql = f"SELECT id FROM {self.table_name}"

        if expr:
            sql += f" WHERE {expr}"

        cursor = self.client.cursor()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
        finally:
            cursor.close()

        return [row[0] for row in rows]

@_handle_exceptions
def create_index(
    connection: Connection,
    vector_store: DB2VS,
    params: Optional[dict[str, Any]] = None,
) -> None:
    """
    Create an index for a Db2 Vector Store.

    This is a thin convenience wrapper that validates the connection and
    delegates to _create_diskann_index, using the table name and
    distance strategy provided by `vector_store`.

    Args:
        connection:
            Open ibm_db_dbi.Connection to the target Db2 database.
        vector_store:
            A DB2VS instance providing table_name (qualified or
            unqualified) and distance_strategy (e.g., EUCLIDEAN or
            EUCLIDEAN_SQUARED).
        params:
            Optional dictionary of index-creation parameters forwarded to
            _create_diskann_index. Expected keys include:
              - "vector_column" (str, required): Name of the vector column.
              - "index_name" (str, required): Name for the new index.
              - "schema" (str, optional): Schema containing the table if
                not specified in table_name.
              - "if_exists" (str, optional): One of "error", "skip",
                or "replace" (default: "error").

    Returns:
        None
    """
    if connection is None:
        raise ValueError("Expected an open ibm_db_dbi connection object")

    _create_diskann_index(
        connection,
        vector_store.table_name,
        vector_store.distance_strategy,
        params,
    )
    return

@_handle_exceptions
def _create_diskann_index(
    connection: Connection,
    table_name: str,
    distance_strategy: DistanceStrategy,
    params: Optional[dict[str, Any]] = None,
) -> None:
    """
    Create a DiskANN vector index on a specified table and vector column.

    The function safely quotes identifiers, resolves schema (from the qualified
    table name, explicit "schema" param, or CURRENT SCHEMA), checks for
    existing indexes with the same name, and conditionally skips, replaces,
    or errors based on "if_exists". After index creation, it runs
    RUNSTATS ... FOR INDEXES ALL to help the optimizer use the new index.

    Args:
        connection:
            Open ibm_db_dbi.Connection to the target Db2 database.
        table_name:
            Target table name. May be qualified (e.g., "SCHEMA.TABLE") or
            unqualified (schema resolved via params or CURRENT SCHEMA).
        distance_strategy:
            Distance function for the index. Supported values map to:
            "EUCLIDEAN" or "EUCLIDEAN_SQUARED".
        params:
            Optional dictionary with index-creation options:
              - "vector_column" (str, required): Name of the vector column
                to index.
              - "index_name" (str, required): Name for the index.
              - "schema" (str, optional): Schema containing the table if
                not included in table_name.
              - "if_exists" (str, optional): Behavior when the index already
                exists. One of:
                  * "error" (default) - raise an error.
                  * "skip" - do nothing if the index exists.
                  * "replace" - drop the existing index and recreate it.

    Returns:
        None
    """
    params = params or {}

    def _quote_ident_and_capitalize(ident: str) -> str:
        """
        Return a safely quoted, uppercase identifier.

        Converts the input identifier to uppercase, escapes internal double quotes
        by doubling them, and wraps the result in double quotes. This ensures the
        identifier is safe for use in Db2 DDL/DML statements.

        Args:
            ident:
                Non-empty identifier string to be normalized and quoted.

        Returns:
            A double-quoted, uppercase, and escape-safe identifier string.

        Raises:
            ValueError:
                If `ident` is None or an empty/whitespace-only string.

        Notes:
            - This does not validate identifier length or reserved words.
            - Case normalization to uppercase matches default semantics for
            unquoted identifiers.
        """
        if ident is None or ident.strip() == "":
            raise ValueError("Identifier must be a non-empty string")
        # Capitalize and escape internal quotes
        return '"' + ident.upper().replace('"', '""') + '"'

    def _parse_qualified_name(name: str, explicit_schema: Optional[str]) -> tuple[str, str]:
        """
        Return (schema, table) from a possibly qualified table name.

        If name is qualified as "SCHEMA.TABLE", it is split at the first "."
        and any surrounding double quotes are stripped. Otherwise, the schema is
        resolved from `explicit_schema` if provided, or via VALUES CURRENT SCHEMA
        as a fallback.

        Args:
            name:
                Table name, optionally qualified (e.g., 'MYSCHEMA.MYTABLE').
                Double quotes around parts are allowed and will be stripped.
            explicit_schema:
                Optional schema to use when name is unqualified.

        Returns:
            A tuple `(schema, table)` with quotes removed but original letter case
            preserved as provided. Final quoting/casing is applied by callers.

        Raises:
            ValueError:
                If the schema cannot be determined (e.g., name is unqualified and
                CURRENT SCHEMA cannot be retrieved).

        Notes:
            - Performs a small DB roundtrip when resolving CURRENT SCHEMA.
            - Only the *first* dot splits schema and table to support table names
            that may include additional dots when quoted properly.
        """
        # Returns (schema, table)
        if "." in name:
            schema, tbl = name.split(".", 1)
            schema = schema.strip('"')
            tbl = tbl.strip('"')
            return schema, tbl
        if explicit_schema:
            return explicit_schema.strip('"'), name.strip('"')
        # Fallback to CURRENT SCHEMA at runtime via a query
        with connection.cursor() as cur:
            cur.execute("VALUES CURRENT SCHEMA")
            row = cur.fetchone()
            if not row or not row[0]:
                raise ValueError("Unable to determine CURRENT SCHEMA for unqualified table name")
            return str(row[0]), name.strip('"')

    def _metric_from_strategy(strategy: DistanceStrategy) -> str:
        """
        Map a DistanceStrategy to the engine's DISTANCE token.

        Converts the provided strategy (by .name if present, else str(strategy))
        to an uppercase token compatible with the DiskANN WITH DISTANCE clause.

        Supported mappings (case-insensitive):
        - 'EUCLIDEAN', 'L2', 'EUCLIDEAN_DISTANCE'           -> 'EUCLIDEAN'
        - 'EUCLIDEAN_SQUARED', 'EUCLIDEAN_SQUARED_DISTANCE',
          'SQUARED_EUCLIDEAN', 'L2_SQUARED', 'L2SQUARED',
          'SQEUCLIDEAN'                                     -> 'EUCLIDEAN_SQUARED'

        Args:
            strategy:
                Distance strategy enum or object whose name identifies the metric.

        Returns:
            One of: 'EUCLIDEAN' or 'EUCLIDEAN_SQUARED'.

        Raises:
            ValueError:
                If the strategy is not one of the supported values/aliases.

        Notes:
            - Aliases are provided for convenience and are normalized internally.
        """
        name = getattr(strategy, "name", str(strategy)).upper()

        # Common aliases supported here for convenience:
        METRIC_TOKEN_MAP = {
            # Euclidean
            "EUCLIDEAN": "EUCLIDEAN",
            "L2": "EUCLIDEAN",
            "EUCLIDEAN_DISTANCE": "EUCLIDEAN",

            # Squared Euclidean
            "EUCLIDEAN_SQUARED": "EUCLIDEAN_SQUARED",
            "EUCLIDEAN_SQUARED_DISTANCE": "EUCLIDEAN_SQUARED",
            "SQUARED_EUCLIDEAN": "EUCLIDEAN_SQUARED",
            "L2_SQUARED": "EUCLIDEAN_SQUARED",
            "L2SQUARED": "EUCLIDEAN_SQUARED",
            "SQEUCLIDEAN": "EUCLIDEAN_SQUARED",
        }

        token = METRIC_TOKEN_MAP.get(name)
        if token:
            return token

        # Keep the error message explicit about what's supported.
        raise ValueError(
            f"Unsupported distance strategy '{name}'. "
            "Supported: EUCLIDEAN/L2, EUCLIDEAN_SQUARED."
        )

    def _exists_index(schema: str, index_name: str) -> bool:
        """
        Return whether an index already exists in the target schema.

        Looks up SYSCAT.INDEXES using a parameterized query and returns True if
        an index with the given (schema, index_name) is found.

        Args:
            schema:
                Schema name to search (case-insensitive; compared uppercase).
            index_name:
                Index name to search (case-insensitive; compared uppercase).

        Returns:
            True if the index exists; False otherwise.

        Notes:
            - Performs a single-row lookup via FETCH FIRST 1 ROW ONLY.
            - Uses uppercase comparison to match Db2's catalog conventions.
        """
        sql = """
            SELECT 1
              FROM SYSCAT.INDEXES
             WHERE INDSCHEMA = ?
               AND INDNAME   = ?
             FETCH FIRST 1 ROW ONLY
        """
        with connection.cursor() as cur:
            cur.execute(sql, (schema.upper(), index_name.upper()))
            return cur.fetchone() is not None

    def _drop_index(schema: str, index_name: str) -> None:
        """
        Drop an existing index using fully qualified and safely quoted names.

        Constructs and executes DROP INDEX "SCHEMA"."INDEX" using the same
        quoting rules as other DDL, ensuring safety against special characters
        and case sensitivity.

        Args:
            schema:
                Schema containing the index to drop.
            index_name:
                Name of the index to drop.

        Returns:
            None

        Raises:
            Exception:
                Any database error raised by the underlying cursor execution.

        Notes:
            - This function does not check existence; callers are expected to ensure
            the index exists or handle errors appropriately.
        """
        ddl = f'DROP INDEX {_quote_ident_and_capitalize(schema)}.{_quote_ident_and_capitalize(index_name)}'
        with connection.cursor() as cur:
            cur.execute(ddl)

    # Extract & validate inputs
    vector_column = params.get("vector_column")
    if not vector_column or not isinstance(vector_column, str):
        raise ValueError("params['vector_column'] (str) is required")

    explicit_schema = params.get("schema")
    schema, table = _parse_qualified_name(table_name, explicit_schema)

    index_name = params.get("index_name")
    if not index_name or not isinstance(index_name, str):
        raise ValueError("params['index_name'] (str) is required")

    if_exists = (params.get("if_exists") or "error").lower()
    if if_exists not in ("error", "skip", "replace"):
        raise ValueError("params['if_exists'] must be one of: 'error', 'skip', 'replace'")

    metric = _metric_from_strategy(distance_strategy)

    fq_table = f'{_quote_ident_and_capitalize(schema)}.{_quote_ident_and_capitalize(table)}'
    fq_index = f'{_quote_ident_and_capitalize(schema)}.{_quote_ident_and_capitalize(index_name)}'
    q_column = _quote_ident_and_capitalize(vector_column)

    # Existence handling
    already = _exists_index(schema, index_name)
    if already:
        if if_exists == "skip":
            return
        if if_exists == "replace":
            _drop_index(schema, index_name)
        else:  # "error"
            raise ValueError(f"Index {schema}.{index_name} already exists")

    # Create the index
    ddl = (
        f"CREATE VECTOR INDEX {fq_index} ON {fq_table} ({q_column}) "
        f"WITH DISTANCE {metric}"
    )

    with connection.cursor() as cur:
        cur.execute(ddl)

    try:
        connection.commit()
    except Exception:
        raise