from unittest.mock import MagicMock, call

import pytest
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from langchain_db2.db2vs import DB2VS, drop_index


def test_init() -> None:
    """Test that the DB2VS class can be initialized."""
    client = MagicMock()
    embedding = DeterministicFakeEmbedding(size=100)
    table_name = "foo"
    db2vs = DB2VS(embedding, table_name, client)
    assert db2vs is not None
    assert isinstance(db2vs, DB2VS)
    assert len(client.mock_calls) == 3


# ---------------------------------------------------------------------------
# create_index - DDL generation (DiskANN, Db2 12.1)
# ---------------------------------------------------------------------------


def _make_db2vs(
    distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
) -> tuple[DB2VS, MagicMock]:
    client = MagicMock()
    embedding = DeterministicFakeEmbedding(size=8)
    db2vs = DB2VS(embedding, "test_table", client, distance_strategy=distance_strategy)
    # Reset call history so we can inspect only create_index calls
    client.reset_mock()
    return db2vs, client


def test_create_index_default_diskann() -> None:
    """create_index with no optional params issues the minimal DiskANN DDL."""
    db2vs, client = _make_db2vs()
    cursor = client.cursor.return_value

    db2vs.create_index("VIDX1")

    executed = cursor.execute.call_args_list
    ddl = executed[0][0][0]
    assert "CREATE VECTOR INDEX VIDX1 ON test_table (embedding)" in ddl
    assert "WITH DISTANCE EUCLIDEAN" in ddl
    # Optional clauses must NOT appear
    assert "ACCURACY" not in ddl
    assert "PARALLEL" not in ddl
    assert "PARAMETERS" not in ddl
    # COMMIT must follow
    assert executed[1] == call("COMMIT")


def test_create_index_with_accuracy_and_parallel() -> None:
    """create_index with accuracy + parallel appends the correct clauses."""
    db2vs, client = _make_db2vs(DistanceStrategy.EUCLIDEAN_DISTANCE)
    cursor = client.cursor.return_value

    db2vs.create_index("VIDX2", accuracy=97, parallel=16)

    ddl = cursor.execute.call_args_list[0][0][0]
    assert "WITH DISTANCE EUCLIDEAN" in ddl
    assert "WITH TARGET ACCURACY 97" in ddl
    assert "BUILD_PARALLELISM 16" in ddl
    assert "PARAMETERS" not in ddl


def test_create_index_with_power_user_params() -> None:
    """create_index with neighbors + ef_construction appends PARAMETERS clause."""
    db2vs, client = _make_db2vs(DistanceStrategy.EUCLIDEAN_DISTANCE)
    cursor = client.cursor.return_value

    db2vs.create_index("VIDX3", neighbors=64, ef_construction=100)

    ddl = cursor.execute.call_args_list[0][0][0]
    assert "WITH DISTANCE EUCLIDEAN" in ddl
    assert "PARAMETERS (NEIGHBORS 64 EFCONSTRUCTION 100)" in ddl
    assert "ACCURACY" not in ddl


def test_create_index_cosine_succeeds() -> None:
    """create_index with COSINE issues the correct DDL (COSINE is supported)."""
    db2vs, client = _make_db2vs(DistanceStrategy.COSINE)
    cursor = client.cursor.return_value

    db2vs.create_index("VIDX_COSINE")

    ddl = cursor.execute.call_args_list[0][0][0]
    assert "CREATE VECTOR INDEX VIDX_COSINE" in ddl
    assert "WITH DISTANCE COSINE" in ddl


def test_create_index_dot_product_raises() -> None:
    """create_index raises ValueError for DOT_PRODUCT — not a valid index distance."""
    db2vs, _ = _make_db2vs(DistanceStrategy.DOT_PRODUCT)
    with pytest.raises((ValueError, RuntimeError)):
        db2vs.create_index("BAD_IDX")


def test_create_index_accuracy_and_power_params_raises() -> None:
    """create_index raises ValueError when accuracy and power params are mixed."""
    db2vs, _ = _make_db2vs()
    with pytest.raises((ValueError, RuntimeError)):
        db2vs.create_index("BAD_IDX", accuracy=90, neighbors=32, ef_construction=64)


def test_create_index_partial_power_params_raises() -> None:
    """create_index raises ValueError for partial power params."""
    db2vs, _ = _make_db2vs()
    with pytest.raises((ValueError, RuntimeError)):
        db2vs.create_index("BAD_IDX", neighbors=32)  # missing ef_construction


# ---------------------------------------------------------------------------
# drop_index
# ---------------------------------------------------------------------------


def test_drop_index_executes_drop_and_commit() -> None:
    """drop_index issues DROP INDEX … and COMMIT when the index exists."""
    client = MagicMock()
    cursor = client.cursor.return_value

    drop_index(client, "HNSW_IDX1")

    executed = [c[0][0] for c in cursor.execute.call_args_list]
    assert executed[0] == "DROP INDEX HNSW_IDX1"
    assert executed[1] == "COMMIT"


def test_drop_index_silently_ignores_missing_index() -> None:
    """drop_index does not raise when SQL0204N (not found) is returned."""
    client = MagicMock()
    cursor = client.cursor.return_value
    cursor.execute.side_effect = [Exception("SQL0204N index not found"), None]

    # Should not raise
    drop_index(client, "NONEXISTENT_IDX")
