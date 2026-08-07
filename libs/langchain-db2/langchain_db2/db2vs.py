"""DB2 vector store wrapper."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import uuid
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
)

import ibm_db_dbi  # type: ignore[import-untyped]
import numpy as np
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from typing_extensions import override

from langchain_db2.utils import EmbeddingsSchema

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ibm_db_dbi import Connection
    from langchain_core.embeddings import Embeddings

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
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        try:
            return func(*args, **kwargs)
        except RuntimeError as db_err:
            # Handle a known type of error (e.g., DB-related) specifically
            exception_msg = "DB-related error occurred."
            logger.exception(exception_msg)
            error_msg = f"Failed due to a DB issue: {db_err}"
            raise RuntimeError(error_msg) from db_err
        except ValueError as val_err:
            # Handle another known type of error specifically
            exception_msg = "Validation error."
            logger.exception(exception_msg)
            error_msg = f"Validation failed: {val_err}"
            raise ValueError(error_msg) from val_err
        except Exception as e:
            # Generic handler for all other exceptions
            exception_msg = f"An unexpected error occurred: {e}"
            logger.exception(exception_msg)
            error_msg = f"Unexpected error: {e}"
            raise RuntimeError(error_msg) from e

    return cast("T", wrapper)


def _table_exists(client: Connection, table_name: str) -> bool:
    cursor = client.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")  # noqa: S608
    except Exception as ex:
        if "SQL0204N" in str(ex):
            return False
        raise
    finally:
        cursor.close()
    return True


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
    error_msg = f"Unsupported distance strategy: {distance_strategy}"
    raise ValueError(error_msg)


@_handle_exceptions
def _create_table(
    client: Connection,
    table_name: str,
    embedding_dim: int,
    tablespace: str | None = None,
    text_field: str = "text",
    id_field: str = "id",
    metadata_field: str = "metadata",
    embedding_field: str = "embedding",
) -> None:
    cols_dict = {
        id_field: "CHAR(16) PRIMARY KEY NOT NULL",
        text_field: "CLOB",
        metadata_field: "BLOB",
        embedding_field: f"vector({embedding_dim}, FLOAT32)",
    }

    if not _table_exists(client, table_name):
        cursor = client.cursor()
        ddl_body = ", ".join(
            f"{col_name} {col_type}" for col_name, col_type in cols_dict.items()
        )
        # VECTOR columns require a large-enough page-size tablespace.
        # Append IN <tablespace> only when the caller supplies one explicitly.
        ddl = f"CREATE TABLE {table_name} ({ddl_body})"
        if tablespace:
            ddl += f" IN {tablespace}"
        try:
            cursor.execute(ddl)
            cursor.execute("COMMIT")
            info_msg = f"Table {table_name} created successfully..."
            logger.info(info_msg)
        finally:
            cursor.close()
    else:
        info_msg = f"Table {table_name} already exists..."
        logger.info(info_msg)


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
            info_msg = f"Table {table_name} dropped successfully..."
            logger.info(info_msg)
        finally:
            cursor.close()
    else:
        info_msg = f"Table {table_name} not found..."
        logger.info(info_msg)


@_handle_exceptions
def drop_index(client: Connection, index_name: str) -> None:
    """Drop a vector index from the database if it exists.

    Args:
        client: The `ibm_db_dbi` connection object
        index_name: The name of the index to drop

    Raises:
        RuntimeError: If an error occurs while dropping the index

    ??? example "Example"

        ```python
        from langchain_db2.db2vs import drop_index

        drop_index(
            client=db_client,  # ibm_db_dbi.Connection
            index_name="VIDX1",
        )
        ```

    """
    cursor = client.cursor()
    try:
        cursor.execute(f"DROP INDEX {index_name}")
        cursor.execute("COMMIT")
        info_msg = f"Index {index_name} dropped successfully..."
        logger.info(info_msg)
    except Exception as ex:
        if "SQL0204N" in str(ex):
            info_msg = f"Index {index_name} not found..."
            logger.info(info_msg)
        else:
            raise
    finally:
        cursor.close()


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
        info_msg = f"Table {table_name} not found…"
        logger.info(info_msg)
        return

    cursor = client.cursor()
    ddl = f"TRUNCATE TABLE {table_name} IMMEDIATE"
    try:
        client.commit()
        cursor.execute(ddl)
        client.commit()
        info_msg = f"Table {table_name} cleared successfully."
        logger.info(info_msg)
    except Exception:
        client.rollback()
        exception_msg = f"Failed to clear table {table_name}. Rolled back."
        logger.exception(exception_msg)
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
        embedding_function: Callable[[str], list[float]] | Embeddings,
        table_name: str,
        client: Connection | None = None,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        query: str | None = "What is a Db2 database",
        params: dict[str, Any] | None = None,
        connection_args: dict[str, Any] | None = None,
        text_field: str = "text",
        id_field: str = "id",
        metadata_field: str = "metadata",
        embedding_field: str = "embedding",
        tablespace: str | None = None,
    ):
        """`DB2VS` vector store.

        Args:
            tablespace: Name of a Db2 tablespace to use when creating the
                vector table.  The tablespace must have a page size large
                enough to hold a full row containing the VECTOR column
                (e.g. a 768-dim FLOAT32 vector occupies 3072 bytes, so any
                page size ≥ 8K works).  When ``None`` (default) no ``IN``
                clause is added and Db2 uses the user's default tablespace.
                Specify this when the default tablespace has a 4K page size
                and would reject the VECTOR column
                (e.g. ``tablespace="TS32K"``).
        """
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
                    # ssl_cert: path to the server certificate file (.arm/.pem).
                    # Only appended when security is also enabled.
                    ssl_cert = connection_args.get("ssl_cert", "")
                    if ssl_cert:
                        conn_str += f"SSLServerCertificate={ssl_cert};"

                self.client = ibm_db_dbi.connect(conn_str, "", "")
            else:
                error_msg = "No valid connection or connection_args is passed"
                raise ValueError(error_msg)
        else:
            self.client = client
        try:
            if not isinstance(embedding_function, EmbeddingsSchema):
                logger.warning(
                    "`embedding_function` is expected to be an Embeddings "
                    "object, support for passing in a function will soon "
                    "be removed.",
                )
            self.embedding_function = embedding_function
            self.query = query
            embedding_dim = self.get_embedding_dimension()

            self.table_name = table_name
            self.distance_strategy = distance_strategy
            self.params = params
            self._text_field = text_field
            self._id_field = id_field
            self._metadata_field = metadata_field
            self._embedding_field = embedding_field
            self._tablespace = tablespace
            _create_table(
                self.client,
                self.table_name,
                embedding_dim,
                tablespace=self._tablespace,
                text_field=self._text_field,
                id_field=self._id_field,
                metadata_field=self._metadata_field,
                embedding_field=self._embedding_field,
            )
        except ibm_db_dbi.DatabaseError as db_err:
            exception_msg = f"Database error occurred while create table: {db_err}"
            logger.exception(exception_msg)
            error_msg = "Failed to create table due to a database error."
            raise RuntimeError(error_msg) from db_err
        except ValueError as val_err:
            exception_msg = f"Validation error: {val_err}"
            logger.exception(exception_msg)
            error_msg = "Failed to create table due to a validation error."
            raise RuntimeError(error_msg) from val_err
        except Exception as ex:
            exception_msg = "An unexpected error occurred while creating the table."
            logger.exception(exception_msg)
            error_msg = "Failed to create table due to an unexpected error."
            raise RuntimeError(error_msg) from ex

    @property
    def embeddings(self) -> Embeddings | None:
        """A property that returns an Embeddings instance.

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
        """Embed the single document by wrapping it in a list."""
        embedded_document = self._embed_documents(
            [self.query if self.query is not None else ""],
        )

        # Get the first (and only) embedding's dimension
        return len(embedded_document[0])

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        if isinstance(self.embedding_function, EmbeddingsSchema):
            return self.embedding_function.embed_documents(texts)
        if callable(self.embedding_function):
            return [self.embedding_function(text) for text in texts]
        error_msg = "The embedding_function is neither Embeddings nor callable."  # type: ignore[unreachable]
        raise TypeError(error_msg)

    def _embed_query(self, text: str) -> list[float]:
        if isinstance(self.embedding_function, EmbeddingsSchema):
            return self.embedding_function.embed_query(text)
        return self.embedding_function(text)

    @_handle_exceptions
    @override
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[Any, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
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
                            .upper(),
                        )
                    else:
                        processed_ids.append(
                            hashlib.sha256(str(uuid.uuid4()).encode())
                            .hexdigest()[:16]
                            .upper(),
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

        # Derive dimension directly from the already-computed embeddings — no
        # second round-trip to the embedding model.
        embedding_len = len(embeddings[0]) if embeddings else self.get_embedding_dimension()
        docs: list[tuple[Any, Any, Any, Any]]
        docs = [
            (id_, f"{embedding}", json.dumps(metadata), text)
            for id_, embedding, metadata, text in zip(
                processed_ids,
                embeddings,
                metadatas,
                texts,
                strict=False,
            )
        ]

        sql_insert = (
            f"INSERT INTO "  # noqa: S608
            f"{self.table_name} ({self._id_field}, {self._embedding_field}, "
            f"{self._metadata_field}, {self._text_field}) "
            f"VALUES (?, VECTOR(cast(? as CLOB(100000)), {embedding_len}, FLOAT32), "
            f"SYSTOOLS.JSON2BSON(?), ?)"
        )

        cursor = self.client.cursor()
        try:
            cursor.executemany(sql_insert, docs)
            cursor.execute("COMMIT")
        finally:
            cursor.close()
        return processed_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: The natural-language text to search for
            k: Number of Documents to return
            filter: Filter by metadata
            kwargs: Additional keyword args

        Returns:
            Documents most similar to a query
        """
        embedding = self._embed_query(query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[Document]:
        """Return documents most similar to a query embedding.

        Args:
            embedding: Embedding to look up documents similar to
            k: Number of Documents to return
            filter: Filter by metadata
            kwargs: Additional keyword args

        Returns:
            Documents ordered from most to least similar
        """
        docs_and_scores = self.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return the top-k documents most similar to a text query, with scores.

        Args:
            query: Natural-language query to embed and search with
            k: Number of results to return
            filter: Filter by metadata
            kwargs: Additional keyword args

        Returns:
            A list of (document, score) pairs ordered by similarity.
                The score is the vector **distance**; lower values indicate
                closer matches.
        """
        embedding = self._embed_query(query)
        return self.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding,
            k=k,
            filter=filter,
            **kwargs,
        )

    @_handle_exceptions
    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[Document, float]]:
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

        sql = f"""
        SELECT {self._id_field},
          {self._text_field},
          SYSTOOLS.BSON2JSON({self._metadata_field}),
          vector_distance({self._embedding_field}, VECTOR('{embedding}', {embedding_len}, FLOAT32),
          {_get_distance_function(self.distance_strategy)}) as distance
        FROM {self.table_name}
        ORDER BY distance
        FETCH FIRST {k} ROWS ONLY
        """  # noqa: S608
        # TODO: No APPROX in "FETCH APPROX FIRST" now. This will be added once
        #  approximate nearest neighbors search in db2 is implemented.

        # Execute the query
        cursor = self.client.cursor()
        try:
            cursor.execute(sql)
            results = cursor.fetchall()

            # Filter results if filter is provided.
            # filter values must be lists, e.g. {"source": ["wiki", "news"]}.
            for result in results:
                metadata = json.loads(result[2] if result[2] is not None else "{}")

                # Apply filtering based on the 'filter' dictionary
                if filter:
                    if all(
                        metadata.get(key) in value
                        for key, value in filter.items()
                        if isinstance(value, list)
                    ):
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
        embedding: list[float],
        k: int,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[Document, float, np.ndarray]]:
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

        sql = f"""
        SELECT {self._id_field},
          {self._text_field},
          SYSTOOLS.BSON2JSON({self._metadata_field}),
          vector_distance({self._embedding_field}, VECTOR('{embedding}', {embedding_len}, FLOAT32),
          {_get_distance_function(self.distance_strategy)}) as distance,
          {self._embedding_field}
        FROM {self.table_name}
        ORDER BY distance
        FETCH FIRST {k} ROWS ONLY
        """  # noqa: S608
        # TODO: No APPROX in "FETCH APPROX FIRST" now. This will be added once
        #  approximate nearest neighbors search in db2 is implemented.

        # Execute the query
        cursor = self.client.cursor()
        try:
            cursor.execute(sql)
            results = cursor.fetchall()

            for result in results:
                # Skip rows whose embedding column is NULL — passing a zero-
                # length array into maximal_marginal_relevance causes a NumPy
                # shape mismatch crash.
                if result[4] is None:
                    continue

                page_content_str = result[1] if result[1] is not None else ""
                metadata = json.loads(result[2] if result[2] is not None else "{}")

                # Apply filter if provided and matches; otherwise, add all
                # documents.
                # filter values must be lists, e.g. {"source": ["wiki"]}.
                if not filter or all(
                    metadata.get(key) in value
                    for key, value in filter.items()
                    if isinstance(value, list)
                ):
                    document = Document(
                        page_content=page_content_str,
                        metadata=metadata,
                    )
                    distance = result[3]
                    current_embedding = np.array(
                        json.loads(result[4]), dtype=np.float32
                    )
                    documents.append((document, distance, current_embedding))
        finally:
            cursor.close()
        return documents

    @_handle_exceptions
    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: list[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[Document, float]]:
        """Return docs and their similarity scores selected.

        Return docs and their similarity scores selected using the
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
            embedding,
            fetch_k,
            filter=filter,
        )
        # Assuming documents_with_scores is a list of tuples (Document, score)

        # If you need to split documents and scores for processing (e.g.,
        # for MMR calculation)
        documents, scores, embeddings = (
            zip(*docs_scores_embeddings, strict=False)
            if docs_scores_embeddings
            else ([], [], [])
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
        return [(documents[i], scores[i]) for i in mmr_selected_indices]

    @_handle_exceptions
    @override
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
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
            kwargs: Additional keyword args

        Returns:
            List of Documents selected by maximal marginal relevance
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )
        return [doc for doc, _ in docs_and_scores]

    @_handle_exceptions
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[Document]:
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
            kwargs: Additional keyword args

        Returns:
            List of Documents selected by maximal marginal relevance

        `max_marginal_relevance_search` requires that `query` returns matched
        embeddings alongside the match documents.
        """
        embedding = self._embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    @_handle_exceptions
    @override
    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete
            kwargs: Additional keyword args
        """
        if ids is None:
            error_msg = "No ids provided to delete."
            raise ValueError(error_msg)

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

        ddl = f"DELETE FROM {self.table_name} WHERE {self._id_field} IN ({placeholders})"  # noqa: S608
        cursor = self.client.cursor()
        try:
            cursor.execute(ddl, hashed_ids)
            cursor.execute("COMMIT")
        finally:
            cursor.close()

    @classmethod
    @_handle_exceptions
    def from_texts(
        cls: type[DB2VS],
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> DB2VS:
        """Return VectorStore initialized from texts and embeddings.

        Args:
            texts: Iterable of strings to add to the vectorstore
            embedding: Embedding to look up documents similar to
            metadatas: Optional list of metadatas associated with the texts
            kwargs: Additional keyword args

        Returns:
            A ready-to-use vector store with the provided texts loaded
        """
        client = kwargs.get("client")
        if client is None:
            error_msg = "client parameter is required..."
            raise ValueError(error_msg)
        params = kwargs.get("params", {})

        table_name = str(kwargs.get("table_name", "langchain"))

        # Default to EUCLIDEAN_DISTANCE when not supplied, matching __init__.
        # Previously this raised TypeError for any caller that omitted the arg.
        distance_strategy = kwargs.get(
            "distance_strategy", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        if not isinstance(distance_strategy, DistanceStrategy):
            error_msg = (
                f"Expected DistanceStrategy, got {type(distance_strategy).__name__}"
            )
            raise TypeError(error_msg)

        query = kwargs.get("query", "What is a Db2 database")

        drop_table(client, table_name)

        vss = cls(
            client=client,
            embedding_function=embedding,
            table_name=table_name,
            distance_strategy=distance_strategy,
            query=query,
            params=params,
            tablespace=kwargs.get("tablespace"),
        )
        vss.add_texts(texts=list(texts), metadatas=metadatas)
        return vss

    @_handle_exceptions
    def get_pks(self, expr: str | None = None) -> list[str]:
        """Get primary keys, optionally filtered by expr.

        Args:
            expr: SQL boolean expression to filter rows, e.g.:
                `id IN ('ABC123','DEF456')` or `title LIKE 'Abc%'`.
                If None, returns all rows.

        Returns:
            List of matching primary-key values.
        """
        sql = f"SELECT {self._id_field} FROM {self.table_name}"  # noqa: S608

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
        self,
        index_name: str,
        accuracy: int | None = None,
        parallel: int | None = None,
        neighbors: int | None = None,
        ef_construction: int | None = None,
    ) -> None:
        """Create a DiskANN vector index on the embedding column for ANN search.

        Db2 12.1 AI Vector Search uses the ``CREATE VECTOR INDEX`` DDL backed
        by the DiskANN algorithm.  Only ``EUCLIDEAN`` and
        ``EUCLIDEAN_SQUARED`` distance metrics are supported for indexing;
        ``COSINE`` and ``DOT_PRODUCT`` work for ``VECTOR_DISTANCE`` queries
        but cannot be used to build an index.

        Args:
            index_name: Name for the new index (e.g. ``"VIDX1"``).
            accuracy: Target accuracy specification (integer 1-100).
                Mutually exclusive with ``neighbors`` / ``ef_construction``.
                If omitted, Db2 uses its default accuracy.
            parallel: Number of parallel workers to use during index build.
                If omitted, Db2 uses its default.
            neighbors: DiskANN power-user parameter ``NEIGHBORS``.  Controls
                the maximum number of bi-directional links per node.  Must be
                provided together with ``ef_construction`` when used.
            ef_construction: DiskANN power-user parameter ``EFCONSTRUCTION``.
                Controls the size of the dynamic candidate list during index
                construction.  Must be provided together with ``neighbors``
                when used.

        Raises:
            ValueError: If ``distance_strategy`` is not ``EUCLIDEAN_DISTANCE``
                or ``MAX_INNER_PRODUCT`` (i.e. not indexable), or if
                ``accuracy`` is combined with
                ``neighbors``/``ef_construction``, or if only one of
                ``neighbors``/``ef_construction`` is given.
            RuntimeError: If a DB2 error occurs while creating the index.

        ??? example "Example - default DiskANN index"

            ```python
            db2vs.create_index("VIDX1")
            ```

        ??? example "Example - with target accuracy and parallel workers"

            ```python
            db2vs.create_index(
                "VIDX2",
                accuracy=97,
                parallel=16,
            )
            ```

        ??? example "Example - with DiskANN power-user parameters"

            ```python
            db2vs.create_index(
                "VIDX3",
                neighbors=64,
                ef_construction=100,
            )
            ```

        """
        # DiskANN only supports EUCLIDEAN / EUCLIDEAN_SQUARED
        _INDEXABLE = {
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            DistanceStrategy.MAX_INNER_PRODUCT,
        }
        if self.distance_strategy not in _INDEXABLE:
            error_msg = (
                f"distance_strategy '{self.distance_strategy}' cannot be used to "
                "build a DiskANN vector index. Only EUCLIDEAN_DISTANCE and "
                "MAX_INNER_PRODUCT (EUCLIDEAN_SQUARED) are supported."
            )
            raise ValueError(error_msg)

        has_accuracy = accuracy is not None
        has_power = neighbors is not None or ef_construction is not None

        if has_accuracy and has_power:
            error_msg = (
                "'accuracy' cannot be combined with 'neighbors' / 'ef_construction'. "
                "Use either target accuracy or power-user parameters, not both."
            )
            raise ValueError(error_msg)

        if (neighbors is None) != (ef_construction is None):
            error_msg = (
                "'neighbors' and 'ef_construction' must be specified together."
            )
            raise ValueError(error_msg)

        distance_func = _get_distance_function(self.distance_strategy)
        ddl = (
            f"CREATE VECTOR INDEX {index_name} ON {self.table_name} "
            f"({self._embedding_field}) WITH DISTANCE {distance_func}"
        )

        if has_accuracy:
            ddl += f" WITH TARGET ACCURACY {accuracy}"

        if parallel is not None:
            ddl += f" PARALLEL {parallel}"

        if has_power:
            ddl += (
                f" PARAMETERS (NEIGHBORS {neighbors}"
                f" EFCONSTRUCTION {ef_construction})"
            )

        cursor = self.client.cursor()
        try:
            cursor.execute(ddl)
            cursor.execute("COMMIT")
            info_msg = f"Index {index_name} created successfully on {self.table_name}."
            logger.info(info_msg)
        finally:
            cursor.close()
