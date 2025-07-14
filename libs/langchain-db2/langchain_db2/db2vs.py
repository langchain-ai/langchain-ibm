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


def _table_exists(client: Connection, table_name: str) -> bool:
    cursor = client.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
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
    raise ValueError(f"Unsupported distance strategy: {distance_strategy}")


@_handle_exceptions
def _create_table(client: Connection, table_name: str, embedding_dim: int) -> None:
    cols_dict = {
        "id": "CHAR(16) PRIMARY KEY NOT NULL",
        "text": "CLOB",
        "metadata": "BLOB",
        "embedding": f"vector({embedding_dim}, FLOAT32)",
    }

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
        client: The ibm_db_dbi connection object.
        table_name: The name of the table to drop.

    Raises:
        RuntimeError: If an error occurs while dropping the table.
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
        client: The ibm_db_dbi connection object.
        table_name: The name of the table to clear.
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

    To use, you should have:
    - the ``ibm_db`` python package installed
    - a connection to db2 database with vector store feature (v12.1.2+)
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
            _create_table(self.client, self.table_name, embedding_dim)
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
            Optional[Embeddings]: Embeddings instance if embedding_function
            is an instance of Embeddings, otherwise returns None.
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
          texts: Iterable of strings to add to the vectorstore.
          metadatas: Optional list of metadatas associated with the texts.
          ids: Optional list of ids for the texts that are being added to
          the vector store.
          kwargs: vectorstore specific parameters
        Return:
          List of ids from adding the texts into the vectorstore.
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
            f"INSERT INTO {self.table_name} (id, embedding, metadata, text) "
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
            query: str,
            k: int, the number for documents to retrieve
            filter: Optional, the filter to apply
        Return:
            List[Document]: documents most similar to a query
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
        """Return docs most similar to query."""
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
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        docs_and_scores = []
        embedding_len = self.get_embedding_dimension()

        query = f"""
        SELECT id,
          text,
          SYSTOOLS.BSON2JSON(metadata),
          vector_distance(embedding, VECTOR('{embedding}', {embedding_len}, FLOAT32),
          {_get_distance_function(self.distance_strategy)}) as distance
        FROM {self.table_name}
        ORDER BY distance
        FETCH FIRST {k} ROWS ONLY
        """
        # TODO: No APPROX in "FETCH APPROX FIRST" now. This will be added once
        # approximate nearest neighbors search in db2 is implemented.

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
        **kwargs: Any,
    ) -> List[Tuple[Document, float, np.ndarray]]:
        documents = []
        embedding_len = self.get_embedding_dimension()

        query = f"""
        SELECT id,
          text,
          SYSTOOLS.BSON2JSON(metadata),
          vector_distance(embedding, VECTOR('{embedding}', {embedding_len}, FLOAT32),
          {_get_distance_function(self.distance_strategy)}) as distance,
          embedding
        FROM {self.table_name}
        ORDER BY distance
        FETCH FIRST {k} ROWS ONLY
        """
        # TODO: No APPROX in "FETCH APPROX FIRST" now. This will be added once
        # approximate nearest neighbors search in db2 is implemented.

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
          self: An instance of the class
          embedding: Embedding to look up documents similar to.
          k: Number of Documents to return. The default value is 4.
          fetch_k: Number of Documents to fetch before filtering to
                   pass to MMR algorithm.
          filter: (Optional[Dict[str, str]]): Filter by metadata. Defaults
          to None.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       The default value is 0.5.
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
          self: An instance of the class
          embedding: Embedding to look up documents similar to.
          k: Number of Documents to return. Defaults to 4.
          fetch_k: Number of Documents to fetch to pass to MMR algorithm.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       Defaults to 0.5.
          filter: Optional[Dict[str, Any]]
          **kwargs: Any
        Returns:
          List of Documents selected by maximal marginal relevance.
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
          self: An instance of the class
          query: Text to look up documents similar to.
          k: Number of Documents to return. The default value is 4.
          fetch_k: Number of Documents to fetch to pass to MMR algorithm.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       The default value is 0.5.
          filter: Optional[Dict[str, Any]]
          **kwargs
        Returns:
          List of Documents selected by maximal marginal relevance.

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
        """Delete by vector IDs.
        Args:
          self: An instance of the class
          ids: List of ids to delete.
          **kwargs
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
        """Return VectorStore initialized from texts and embeddings."""
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
                  "id IN ('ABC123','DEF456')" or "title LIKE 'Abc%'".
                  If None, returns all rows.
        Returns:
            List[str]: List of matching primary-key values.
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
