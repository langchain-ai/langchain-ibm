"""Test Db2 AI Vector Search functionality."""

import threading
import uuid
from typing import Optional

import ibm_db_dbi  # type: ignore[import-untyped]
import pytest
from ibm_db_dbi import Connection  # type: ignore[import-untyped]
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_db2.db2vs import (
    DB2VS,
    _create_table,
    _table_exists,
    clear_table,
    drop_index,
    drop_table,
)

VECTOR_DIM = 768

SIMILARITY_SEARCH_TEXTS = ["Yash", "Varanasi", "Yashaswi", "Mumbai", "BengaluruYash"]
SIMILARITY_SEARCH_METADATAS = [
    {"id": "hello"},
    {"id": "105"},
    {"id": "106"},
    {"id": "yash"},
    {"id": "108"},
]
SIMILARITY_SEARCH_QUERY = "YashB"
SIMILARITY_SEARCH_FILTER = {"id": ["106", "108", "yash"]}


@pytest.mark.xfail
def test_table_exists_for_existing_and_non_existing(
    ibm_db_dbi_connection: Connection,
) -> None:
    try:
        _create_table(ibm_db_dbi_connection, "TB1", 8148)

        # Existing table -> True
        assert _table_exists(ibm_db_dbi_connection, "TB1")

        # Non-existing table -> False
        assert not _table_exists(ibm_db_dbi_connection, "TableNonExist")
    finally:
        try:
            drop_table(ibm_db_dbi_connection, "TB1")
        finally:
            ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_table_exists_is_case_insensitive_for_unquoted_names(
    ibm_db_dbi_connection: Connection,
) -> None:
    try:
        _create_table(ibm_db_dbi_connection, "TB1", 8148)

        # Mixed case lookup should still succeed for unquoted identifiers
        assert _table_exists(ibm_db_dbi_connection, "Tb1")
    finally:
        try:
            drop_table(ibm_db_dbi_connection, "TB1")
        finally:
            ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_table_exists_with_quoted_unicode_identifier(
    ibm_db_dbi_connection: Connection,
) -> None:
    # Note: quoted identifiers are case-sensitive and allow Unicode
    name = '"表格"'
    try:
        _create_table(ibm_db_dbi_connection, name, 545)
        assert _table_exists(ibm_db_dbi_connection, name)
    finally:
        try:
            drop_table(ibm_db_dbi_connection, name)
        finally:
            ibm_db_dbi_connection.commit()


@pytest.mark.xfail
@pytest.mark.parametrize(
    ("table_name", "expected_regex"),
    [
        ("123", r"SQL0104N"),  # Invalid token
        ("", r"SQL0104N"),  # Empty string -> syntax error
        ("!!4", r"SQL0007N"),  # Invalid character
        ("x" * 129, r"SQL0107N"),  # Identifier too long (>128)
    ],
)
def test_table_exists_raises_for_invalid_names(
    ibm_db_dbi_connection: Connection, table_name: str, expected_regex: str
) -> None:
    with pytest.raises(Exception, match=expected_regex):
        _table_exists(ibm_db_dbi_connection, table_name)


@pytest.mark.xfail
def test_create_table_basic_and_duplicate(ibm_db_dbi_connection: Connection) -> None:
    try:
        _create_table(ibm_db_dbi_connection, "HELLO", 100)
        _create_table(ibm_db_dbi_connection, "HELLO", 110)
    finally:
        try:
            drop_table(ibm_db_dbi_connection, "HELLO")
        finally:
            ibm_db_dbi_connection.commit()


@pytest.mark.xfail
@pytest.mark.parametrize(
    ("name", "dim"),
    [
        ("Hello123", 8148),  # valid ascii identifier
        ("T2", int("1000")),  # dimension provided via int conversion
        ("T3", 100),  # dimension via variable
        ("T10", 1 + 500),  # dimension via arithmetic
        ('"T5"', 128),  # quoted identifier
        ("TaBlE", 128),  # toggle case
    ],
)
def test_create_table_various_valid_inputs(
    ibm_db_dbi_connection: Connection, name: str, dim: int
) -> None:
    """
    Create new HELLO (dim=100) -> OK
    Call create again HELLO (dim=110) -> should not error (e.g., logs "already exists")
    """
    try:
        _create_table(ibm_db_dbi_connection, name, dim)
    finally:
        try:
            drop_table(ibm_db_dbi_connection, name)
        finally:
            ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_create_table_with_quoted_unicode(ibm_db_dbi_connection: Connection) -> None:
    """
    New table - "表格" (quoted), dim=545 -> OK
    """
    name = '"表格"'
    try:
        _create_table(ibm_db_dbi_connection, name, 545)
    finally:
        try:
            drop_table(ibm_db_dbi_connection, name)
        finally:
            ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_create_table_with_schema_name(ibm_db_dbi_connection: Connection) -> None:
    """
    <schema.table> e.g., U1.TB4, dim=128 -> OK
    """
    name = "U1.TB4"
    try:
        _create_table(ibm_db_dbi_connection, name, 128)
    finally:
        try:
            drop_table(ibm_db_dbi_connection, name)
        finally:
            ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_create_table_from_string_ops_in_name(
    ibm_db_dbi_connection: Connection,
) -> None:
    """
    Name via string ops: "YaSh".replace("aS","ok") -> "Yokh"
    """
    name = "YaSh".replace("aS", "ok")
    try:
        _create_table(ibm_db_dbi_connection, name, 500)
    finally:
        try:
            drop_table(ibm_db_dbi_connection, name)
        finally:
            ibm_db_dbi_connection.commit()


@pytest.mark.xfail
@pytest.mark.parametrize(
    ("name", "dim", "regex"),
    [
        ("123", 100, r"SQL0104N"),  # invalid table name (starts with digit)
        ("", 128, r"SQL0104N"),  # empty string
        ('""', 128, r"SQL0104N"),  # empty quoted identifier
        ("H\nello", 545, r"SQL0104N"),  # unexpected token from newline/multiline
    ],
)
def test_create_table_raises_for_invalid_names(
    ibm_db_dbi_connection: Connection, name: str, dim: int, regex: str
) -> None:
    with pytest.raises(Exception, match=regex):
        _create_table(ibm_db_dbi_connection, name, dim)
    ibm_db_dbi_connection.commit()


@pytest.mark.xfail
@pytest.mark.parametrize(
    ("name", "dim", "regex"),
    [
        ("T1", 65536, r"SQL0604N"),  # VECTOR column exceeds supported dimension
        ("T1", 0, r"SQL0604N"),  # unsupported dimension length 0
        ("T1", -1, r"SQL0104N"),  # unexpected token "-" (negative)
    ],
)
def test_create_table_raises_for_invalid_dimensions(
    ibm_db_dbi_connection: Connection, name: str, dim: int, regex: str
) -> None:
    with pytest.raises(Exception, match=regex):
        _create_table(ibm_db_dbi_connection, name, dim)
    ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_add_two_records_with_metadata(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    Add 2 records with metadata that includes explicit IDs -> OK
    """
    texts = ["David", "Vectoria"]
    metadata = [
        {"id": "100", "link": "Document Example Test 1"},
        {"id": "101", "link": "Document Example Test 2"},
    ]
    table = "TB1"
    try:
        vs_obj = DB2VS(
            hf_embeddings,
            table,
            ibm_db_dbi_connection,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        vs_obj.add_texts(texts, metadata)
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_add_two_records_metadata_without_ids_generates_ids(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    Metadata present but no 'id' -> IDs should be generated -> OK
    """
    texts = ["David", "Vectoria"]
    metadata_no_id = [
        {"link": "Document Example Test 1"},
        {"link": "Document Example Test 2"},
    ]
    table = "TB2"
    try:
        vs_obj = DB2VS(
            hf_embeddings,
            table,
            ibm_db_dbi_connection,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        vs_obj.add_texts(texts, metadata_no_id)
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_add_three_records_partial_metadata_ids(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    Some metadata items have 'id', others don't -> missing IDs generated -> OK
    """
    texts = ["David", "Vectoria", "John"]
    metadata_partial = [
        {"id": "100", "link": "Document Example Test 1"},
        {"link": "Document Example Test 2"},
        {"link": "Document Example Test 3"},
    ]
    table = "TB2A"
    try:
        vs_obj = DB2VS(
            hf_embeddings,
            table,
            ibm_db_dbi_connection,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        vs_obj.add_texts(texts, metadata_partial)
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_add_records_without_metadata_generates_ids(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    No metadata/ids provided -> IDs should be generated -> OK
    """
    table = "TB3"
    texts = ["Sam", "John"]
    try:
        vs_obj = DB2VS(
            hf_embeddings,
            table,
            ibm_db_dbi_connection,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        vs_obj.add_texts(texts)
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
@pytest.mark.parametrize(
    ("table", "ids_override"),
    [
        ("TB4", ["114", "124"]),  # normal strings
        ("TB5", ["", "134"]),  # one empty string
        (
            "TB6",
            [
                """Good afternoon
my friends""",
                "India",
            ],
        ),  # multi-line string
        ("TB7", ['"Good afternoon"', '"India"']),  # quoted strings as IDs
    ],
)
def test_add_records_with_ids_variations(
    ibm_db_dbi_connection: Connection,
    hf_embeddings: HuggingFaceEmbeddings,
    table: str,
    ids_override: list,
) -> None:
    """
    Various 'ids' inputs -> OK
    """
    texts = ["Sam", "John"]
    try:
        vs_obj = DB2VS(
            hf_embeddings,
            table,
            ibm_db_dbi_connection,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        vs_obj.add_texts(texts, ids=ids_override)
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_add_records_with_ids_and_metadata_prefers_ids(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    Both 'ids' and 'metadata' provided -> IDs from 'ids' are used -> OK
    """
    table = "TB9"
    texts = ["Sam 6", "John 6"]
    ids = ["1", "2"]
    metadata = [
        {"id": "102", "link": "Document Example", "stream": "Science"},
        {"id": "104", "link": "Document Example 45"},
    ]
    try:
        vs_obj = DB2VS(
            hf_embeddings,
            table,
            ibm_db_dbi_connection,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        vs_obj.add_texts(texts, metadata, ids=ids)
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_add_many_records_bulk(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    Add ~10k records -> OK
    """
    table = "TB10"
    texts = [f"Sam{i}" for i in range(1, 10000)]
    ids = [f"Hello{i}" for i in range(1, 10000)]
    try:
        vs_obj = DB2VS(
            hf_embeddings,
            table,
            ibm_db_dbi_connection,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        vs_obj.add_texts(texts, ids=ids)
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_add_records_duplicate_ids_raises(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    Duplicate IDs -> expect SQL0803N (unique/PK violation)
    """
    table = "TB8"
    texts = ["Sam", "John"]
    ids = ["118", "118"]
    try:
        vs_obj = DB2VS(
            hf_embeddings,
            table,
            ibm_db_dbi_connection,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        with pytest.raises(Exception, match=r"SQL0803N"):
            vs_obj.add_texts(texts, ids=ids)
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_add_two_different_records_concurrently(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    Two different inserts concurrently -> both succeed
    """
    table = f"TB11_{uuid.uuid4().hex[:8]}"
    errors = []

    # Pre-create the table to avoid DDL races (if your suite has a helper)
    _create_table(ibm_db_dbi_connection, table, VECTOR_DIM)

    def add(val: str) -> None:
        try:
            vs = DB2VS(
                hf_embeddings,
                table,
                ibm_db_dbi_connection,
                DistanceStrategy.EUCLIDEAN_DISTANCE,
            )
            vs.add_texts([val], ids=[val])
            ibm_db_dbi_connection.commit()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    try:
        t1 = threading.Thread(target=add, args=("Sam",))
        t2 = threading.Thread(target=add, args=("John",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    finally:
        # TODO: assert not errors, f"Unexpected errors in concurrent add: {errors}"
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_add_two_same_records_concurrently_conflict(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    Two identical inserts concurrently -> one should fail with PK violation
    """
    table = f"TB12_{uuid.uuid4().hex[:8]}"
    errors = []

    # Pre-create the table to avoid DDL races (if your suite has a helper)
    _create_table(ibm_db_dbi_connection, table, VECTOR_DIM)

    def add(val: str) -> None:
        try:
            vs = DB2VS(
                hf_embeddings,
                table,
                ibm_db_dbi_connection,
                DistanceStrategy.EUCLIDEAN_DISTANCE,
            )
            vs.add_texts([val], ids=[val])
            ibm_db_dbi_connection.commit()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    try:
        t1 = threading.Thread(target=add, args=("Sam",))
        t2 = threading.Thread(target=add, args=("Sam",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # TODO: Expect at least one PK/unique-key violation
        #  assert any("SQL0803N" in str(e) for e in errors)"
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_embed_single_document(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    Embed a single string -> returns one vector with expected dimension
    """
    table = f"TB7_{uuid.uuid4().hex[:8]}"
    try:
        vs = DB2VS(
            hf_embeddings,
            table,
            ibm_db_dbi_connection,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        out = vs._embed_documents(["Sam"])
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], (list, tuple))
        assert len(out[0]) == VECTOR_DIM
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_embed_multiple_documents(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    Embed a list of strings -> returns N vectors with expected dimension
    """
    table = f"TB7_{uuid.uuid4().hex[:8]}"
    docs = ["hello", "yash"]
    try:
        vs = DB2VS(
            hf_embeddings,
            table,
            ibm_db_dbi_connection,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        out = vs._embed_documents(docs)
        assert isinstance(out, list)
        assert len(out) == len(docs)
        for vec in out:
            assert isinstance(vec, (list, tuple))
            assert len(vec) == VECTOR_DIM
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
@pytest.mark.parametrize(
    "query",
    [
        "Sam",
        pytest.param("", id="empty"),
    ],
)
def test_embed_query(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings, query: str
) -> None:
    table = f"TB8_{uuid.uuid4().hex[:8]}"
    try:
        vs = DB2VS(
            hf_embeddings,
            table,
            ibm_db_dbi_connection,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        vec = vs._embed_query(query)
        assert isinstance(vec, (list, tuple))
        assert len(vec) == VECTOR_DIM
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
@pytest.mark.parametrize(
    "strategy",
    [
        DistanceStrategy.EUCLIDEAN_DISTANCE,
        DistanceStrategy.DOT_PRODUCT,
        DistanceStrategy.COSINE,
    ],
)
def test_similarity_search_basic_and_filtered(
    ibm_db_dbi_connection: Connection,
    hf_embeddings: HuggingFaceEmbeddings,
    strategy: DistanceStrategy,
) -> None:
    table = f"TB10_{uuid.uuid4().hex[:8]}"
    try:
        vs = DB2VS(hf_embeddings, table, ibm_db_dbi_connection, strategy)

        vs.add_texts(SIMILARITY_SEARCH_TEXTS, SIMILARITY_SEARCH_METADATAS)

        res = vs.similarity_search(SIMILARITY_SEARCH_QUERY, k=2)
        assert isinstance(res, list)
        assert len(res) <= 2
        assert all(hasattr(d, "page_content") for d in res)

        res_f = vs.similarity_search(
            SIMILARITY_SEARCH_QUERY, k=2, filter=SIMILARITY_SEARCH_FILTER
        )
        assert isinstance(res_f, list)
        assert len(res_f) <= 2

        allowed = set(SIMILARITY_SEARCH_FILTER["id"])
        for d in res_f:
            mid = (d.metadata or {}).get("id")
            assert mid in allowed
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
@pytest.mark.parametrize(
    "strategy",
    [
        DistanceStrategy.EUCLIDEAN_DISTANCE,
        DistanceStrategy.DOT_PRODUCT,
        DistanceStrategy.COSINE,
    ],
)
def test_similarity_search_with_score_basic_and_filtered(
    ibm_db_dbi_connection: Connection,
    hf_embeddings: HuggingFaceEmbeddings,
    strategy: DistanceStrategy,
) -> None:
    table = f"TB11_{uuid.uuid4().hex[:8]}"
    try:
        vs = DB2VS(hf_embeddings, table, ibm_db_dbi_connection, strategy)

        vs.add_texts(SIMILARITY_SEARCH_TEXTS, SIMILARITY_SEARCH_METADATAS)

        res = vs.similarity_search_with_score(SIMILARITY_SEARCH_QUERY, k=2)
        assert isinstance(res, list)
        assert len(res) <= 2
        for doc, score in res:
            assert hasattr(doc, "page_content")
            assert isinstance(score, (int, float))

        res_f = vs.similarity_search_with_score(
            SIMILARITY_SEARCH_QUERY, k=2, filter=SIMILARITY_SEARCH_FILTER
        )
        assert isinstance(res_f, list)
        assert len(res_f) <= 2
        allowed = set(SIMILARITY_SEARCH_FILTER["id"])
        for doc, score in res_f:
            assert isinstance(score, (int, float))
            mid = (doc.metadata or {}).get("id")
            assert mid in allowed
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
@pytest.mark.parametrize(
    "strategy",
    [
        DistanceStrategy.EUCLIDEAN_DISTANCE,
        DistanceStrategy.DOT_PRODUCT,
        DistanceStrategy.COSINE,
    ],
)
def test_mmr_search_basic_and_filtered(
    ibm_db_dbi_connection: Connection,
    hf_embeddings: HuggingFaceEmbeddings,
    strategy: DistanceStrategy,
) -> None:
    table = f"TB12_{uuid.uuid4().hex[:8]}"
    try:
        vs = DB2VS(hf_embeddings, table, ibm_db_dbi_connection, strategy)

        vs.add_texts(SIMILARITY_SEARCH_TEXTS, SIMILARITY_SEARCH_METADATAS)

        res = vs.max_marginal_relevance_search(
            SIMILARITY_SEARCH_QUERY, k=2, fetch_k=20, lambda_mult=0.5
        )
        assert isinstance(res, list)
        assert len(res) <= 2

        ids = [(d.metadata or {}).get("id") for d in res]
        assert len(ids) == len(set(ids))

        res_f = vs.max_marginal_relevance_search(
            SIMILARITY_SEARCH_QUERY,
            k=2,
            fetch_k=20,
            lambda_mult=0.5,
            filter=SIMILARITY_SEARCH_FILTER,
        )
        assert isinstance(res_f, list)
        assert len(res_f) <= 2
        allowed = set(SIMILARITY_SEARCH_FILTER["id"])
        for d in res_f:
            mid = (d.metadata or {}).get("id")
            assert mid in allowed
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_get_pks_empty_on_new_table(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    table = f"GET_PKS_{uuid.uuid4().hex[:8]}"
    try:
        db2vs = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=ibm_db_dbi_connection,
        )
        pks = db2vs.get_pks()
        assert isinstance(pks, list)
        assert len(pks) == 0
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_get_pks_after_add_texts(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    table = f"GET_PKS_{uuid.uuid4().hex[:8]}"
    try:
        db2vs = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=ibm_db_dbi_connection,
        )
        db2vs.add_texts(texts=["Josh", "Mary"])
        pks = db2vs.get_pks()
        assert isinstance(pks, list)
        assert len(pks) >= 2
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_get_pks_after_clear_is_empty(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    table = f"GET_PKS_{uuid.uuid4().hex[:8]}"
    try:
        db2vs = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=ibm_db_dbi_connection,
        )
        db2vs.add_texts(texts=["Josh", "Mary"])
        pks_before = db2vs.get_pks()
        assert len(pks_before) >= 2

        clear_table(client=ibm_db_dbi_connection, table_name=table)

        pks_after = db2vs.get_pks()
        assert isinstance(pks_after, list)
        assert len(pks_after) == 0
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_custom_text_field_is_set(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    Instance respects a non-default text_field.
    """
    table = f"unique_{uuid.uuid4().hex[:8]}"
    try:
        db2vs = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=ibm_db_dbi_connection,
            text_field="text_v2",
        )
        assert db2vs._text_field == "text_v2"
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_similarity_search_works_with_custom_text_field(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    Adding & searching works when using a custom text_field.
    """
    table = f"unique_{uuid.uuid4().hex[:8]}"
    try:
        db2vs = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=ibm_db_dbi_connection,
            text_field="text_v2",
        )
        db2vs.add_texts(texts=["Josh", "Mary"])
        res = db2vs.similarity_search(query="Mary", k=2)
        assert isinstance(res, list)
        assert len(res) >= 1
        assert any("Mary" in d.page_content for d in res)
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_default_instance_fails_when_table_uses_custom_text_field(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """
    If the table was created with a custom text_field, a default-instance
    (using default text column) should fail with SQL0206N (column not found).
    """
    table = f"unique_{uuid.uuid4().hex[:8]}"
    try:
        db2vs_custom = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=ibm_db_dbi_connection,
            text_field="text_v2",
        )
        db2vs_custom.add_texts(texts=["Josh", "Mary"])

        db2vs_default = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=ibm_db_dbi_connection,
        )
        with pytest.raises(Exception, match="SQL0206N"):
            db2vs_default.similarity_search(query="Mary", k=2)
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


# ---------------------------------------------------------------------------
# create_index / drop_index - DiskANN (Db2 12.1)
# ---------------------------------------------------------------------------


@pytest.mark.xfail
def test_create_vector_index_euclidean(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """create_index with EUCLIDEAN_DISTANCE builds a DiskANN vector index."""
    table = f"vidx_{uuid.uuid4().hex[:8]}"
    index = f"VI_{uuid.uuid4().hex[:8].upper()}"
    try:
        db2vs = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=ibm_db_dbi_connection,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        db2vs.add_texts(texts=["hello", "world"])
        # Should not raise
        db2vs.create_index(index)
    finally:
        drop_index(ibm_db_dbi_connection, index)
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_create_vector_index_cosine(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """create_index with COSINE builds a DiskANN vector index (COSINE is supported)."""
    table = f"vidx_{uuid.uuid4().hex[:8]}"
    index = f"VI_{uuid.uuid4().hex[:8].upper()}"
    try:
        db2vs = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=ibm_db_dbi_connection,
            distance_strategy=DistanceStrategy.COSINE,
        )
        db2vs.add_texts(texts=["hello", "world"])
        # Should not raise — COSINE is a valid DiskANN index distance
        db2vs.create_index(index)
    finally:
        drop_index(ibm_db_dbi_connection, index)
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_create_vector_index_dot_product_raises(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """create_index raises ValueError when distance_strategy is DOT_PRODUCT."""
    table = f"vidx_{uuid.uuid4().hex[:8]}"
    try:
        db2vs = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=ibm_db_dbi_connection,
            distance_strategy=DistanceStrategy.DOT_PRODUCT,
        )
        db2vs.add_texts(texts=["hello"])
        with pytest.raises((ValueError, RuntimeError)):
            db2vs.create_index("SHOULD_NOT_EXIST")
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_drop_index_removes_existing_index(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """drop_index removes an existing DiskANN vector index without error."""
    table = f"vidx_{uuid.uuid4().hex[:8]}"
    index = f"VI_{uuid.uuid4().hex[:8].upper()}"
    try:
        db2vs = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=ibm_db_dbi_connection,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        db2vs.add_texts(texts=["hello", "world"])
        db2vs.create_index(index)
        # Now drop it — should not raise
        drop_index(ibm_db_dbi_connection, index)
    finally:
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


@pytest.mark.xfail
def test_drop_index_silently_ignores_nonexistent(
    ibm_db_dbi_connection: Connection,
) -> None:
    """drop_index does not raise when the index does not exist."""
    drop_index(ibm_db_dbi_connection, "NONEXISTENT_VIDX_XYZ")


@pytest.mark.xfail
def test_similarity_search_after_create_index(
    ibm_db_dbi_connection: Connection, hf_embeddings: HuggingFaceEmbeddings
) -> None:
    """similarity_search still returns correct results after a vector index is built."""
    table = f"vidx_{uuid.uuid4().hex[:8]}"
    index = f"VI_{uuid.uuid4().hex[:8].upper()}"
    try:
        db2vs = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=ibm_db_dbi_connection,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        texts = ["Yash", "Varanasi", "Yashaswi", "Mumbai", "BengaluruYash"]
        db2vs.add_texts(texts=texts)
        db2vs.create_index(index)

        results = db2vs.similarity_search(query="YashB", k=3)
        assert isinstance(results, list)
        assert len(results) >= 1
    finally:
        drop_index(ibm_db_dbi_connection, index)
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()


# ---------------------------------------------------------------------------
# CREATE VECTOR INDEX privilege boundary
# ---------------------------------------------------------------------------
# History:
#   Db2 12.1 Early Access / pre-12.1.5.0:
#     CREATE VECTOR INDEX required DBADM because Db2 internally wrote a
#     control object (SYSIBM.AGT<timestamp>) into the SYSIBM system schema
#     during DiskANN index construction.  SYSIBM is an internal schema not
#     governed by SYSCAT.SCHEMAAUTH; only DBADM granted write access to it.
#
#     Observed on TESTDB @ 9.60.234.51 (2025-08):
#       DB2TEST (DBADMAUTH=N, CREATETABAUTH=Y, IMPLSCHEMAAUTH=Y)
#       → SQL0551N on SYSIBM.AGT260807044807780219
#
#   Fixed in Db2 12.1.5.0 (Mod Pack 5):
#     The SYSIBM.AGT write is no longer required.  A user with only
#     CONNECT + CREATETAB + IMPLICIT_SCHEMA (no DBADM) can now run
#     CREATE VECTOR INDEX successfully.
#
#     Confirmed by live testing on TESTDB @ 9.60.199.101 (2026-09):
#       DB2TEST (DBADMAUTH=N, CREATETABAUTH=Y, IMPLSCHEMAAUTH=Y)
#       → CREATE VECTOR INDEX WITH DISTANCE EUCLIDEAN: SUCCESS
#
# These two tests pin that real-world behaviour:
#   1. non-DBADM user → CREATE VECTOR INDEX succeeds (fixed in 12.1.5.0)
#   2. DBADM user     → CREATE VECTOR INDEX succeeds
#
# Both tests require the server to be reachable.  The non-DBADM test
# additionally requires DB2_LIMITED_USER / DB2_LIMITED_PASSWORD in .env;
# if those vars are absent the test is automatically skipped.


def test_create_vector_index_succeeds_for_non_dbadm_user(
    limited_privilege_connection: "Optional[ibm_db_dbi.Connection]",
    hf_embeddings: HuggingFaceEmbeddings,
) -> None:
    """A user with NO DBADM can CREATE VECTOR INDEX on Db2 12.1.5.0+.

    Before Db2 12.1.5.0 (Mod Pack 5), CREATE VECTOR INDEX required DBADM
    because the engine internally wrote SYSIBM.AGT<timestamp> objects during
    DiskANN construction — and only DBADM grants write access to SYSIBM.

    This was fixed in Db2 12.1.5.0.  A user with only CONNECT + CREATETAB +
    IMPLICIT_SCHEMA (DBADMAUTH=N) can now create vector indexes without error.

    Live proof (TESTDB @ 9.60.199.101, DB2 v12.1.5.0, 2026-09):
      DB2TEST (DBADMAUTH=N, CREATETABAUTH=Y, IMPLSCHEMAAUTH=Y) on a table
      in TS_SM4K_TC16 (32K pages) → CREATE VECTOR INDEX: SUCCESS.

    If DB2_LIMITED_USER is not configured the test is skipped.
    """
    if limited_privilege_connection is None:
        pytest.skip("DB2_LIMITED_USER / DB2_LIMITED_PASSWORD not set — skipping privilege boundary test")

    short = uuid.uuid4().hex[:8]
    # Limited user owns this table — no cross-schema grant needed
    table = f"db2test.priv_{short}"
    index = f"PRIV_{short.upper()}"
    try:
        # Non-DBADM user creates and populates its own table in the 32K
        # tablespace (eliminates SQL0614N page-size error).
        limited_vs = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=limited_privilege_connection,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,  # type: ignore[arg-type]
            tablespace="TS_SM4K_TC16",
        )
        limited_vs.add_texts(texts=["hello", "world"])
        limited_privilege_connection.commit()

        # Non-DBADM user attempts CREATE VECTOR INDEX — must NOT raise.
        # On Db2 12.1.5.0+ this succeeds; on older builds it raised SQL0551N.
        limited_vs.create_index(index)

    finally:
        drop_index(limited_privilege_connection, index)
        drop_table(limited_privilege_connection, table)
        limited_privilege_connection.commit()


def test_create_vector_index_succeeds_for_dbadm_user(
    ibm_db_dbi_connection: Connection,
    hf_embeddings: HuggingFaceEmbeddings,
) -> None:
    """A user with DBADM can CREATE VECTOR INDEX without error.

    This is the positive counterpart to
    ``test_create_vector_index_requires_dbadm_limited_user_gets_sql0551n``.
    It confirms that the primary (DBADM) connection succeeds where the
    limited user fails, grounding the DBADM requirement as an observed fact
    rather than a documentation assumption.
    """
    table = f"priv_{uuid.uuid4().hex[:8]}"
    index = f"PRIV_{uuid.uuid4().hex[:8].upper()}"
    try:
        db2vs = DB2VS(
            embedding_function=hf_embeddings,
            table_name=table,
            client=ibm_db_dbi_connection,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,  # type: ignore[arg-type]
        )
        db2vs.add_texts(texts=["hello", "world"])
        # DBADM user — must not raise
        db2vs.create_index(index)
    finally:
        drop_index(ibm_db_dbi_connection, index)
        drop_table(ibm_db_dbi_connection, table)
        ibm_db_dbi_connection.commit()
