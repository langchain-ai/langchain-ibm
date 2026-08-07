from unittest.mock import MagicMock, call

import pytest
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from langchain_db2.db2vs import DB2VS, _quote_ident, drop_index


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
# _quote_ident
# ---------------------------------------------------------------------------


def test_quote_ident_uppercases_and_wraps() -> None:
    assert _quote_ident("embedding") == '"EMBEDDING"'


def test_quote_ident_escapes_internal_double_quotes() -> None:
    assert _quote_ident('weird"name') == '"WEIRD""NAME"'


def test_quote_ident_empty_raises() -> None:
    with pytest.raises(ValueError):
        _quote_ident("")


def test_quote_ident_whitespace_only_raises() -> None:
    with pytest.raises(ValueError):
        _quote_ident("   ")


# ---------------------------------------------------------------------------
# create_index - DDL generation (DiskANN, Db2 12.1)
# ---------------------------------------------------------------------------


def _make_db2vs(
    distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
    table_name: str = "test_table",
) -> tuple[DB2VS, MagicMock]:
    client = MagicMock()
    # Simulate SYSCAT.INDEXES returning no row (index does not exist yet)
    client.cursor.return_value.fetchone.return_value = None
    embedding = DeterministicFakeEmbedding(size=8)
    db2vs = DB2VS(embedding, table_name, client, distance_strategy=distance_strategy)
    # Reset call history so we can inspect only create_index calls
    client.reset_mock()
    # Ensure fetchone always returns None (index not present) by default
    client.cursor.return_value.fetchone.return_value = None
    return db2vs, client


def _find_ddl(cursor: MagicMock) -> str:
    """Return the CREATE VECTOR INDEX DDL statement from cursor execute calls."""
    for c in cursor.execute.call_args_list:
        stmt = c[0][0]
        if "CREATE VECTOR INDEX" in stmt:
            return stmt
    msg = "No CREATE VECTOR INDEX statement found in cursor calls"
    raise AssertionError(msg)


def test_create_index_default_diskann() -> None:
    """create_index with no optional params issues the minimal DiskANN DDL."""
    db2vs, client = _make_db2vs()
    cursor = client.cursor.return_value

    db2vs.create_index("VIDX1")

    ddl = _find_ddl(cursor)
    # Index name and column must be double-quoted
    assert 'CREATE VECTOR INDEX "VIDX1"' in ddl
    assert '"EMBEDDING"' in ddl
    assert "WITH DISTANCE EUCLIDEAN" in ddl
    # Optional clauses must NOT appear
    assert "ACCURACY" not in ddl
    assert "PARALLELISM" not in ddl
    assert "PARAMETERS" not in ddl


def test_create_index_with_accuracy_and_parallel() -> None:
    """create_index with accuracy + parallel appends the correct clauses."""
    db2vs, client = _make_db2vs(DistanceStrategy.EUCLIDEAN_DISTANCE)
    cursor = client.cursor.return_value

    db2vs.create_index("VIDX2", accuracy=97, parallel=16)

    ddl = _find_ddl(cursor)
    assert "WITH DISTANCE EUCLIDEAN" in ddl
    assert "WITH TARGET ACCURACY 97" in ddl
    assert "BUILD_PARALLELISM 16" in ddl
    assert "PARAMETERS" not in ddl


def test_create_index_with_power_user_params() -> None:
    """create_index with neighbors + ef_construction appends PARAMETERS clause."""
    db2vs, client = _make_db2vs(DistanceStrategy.EUCLIDEAN_DISTANCE)
    cursor = client.cursor.return_value

    db2vs.create_index("VIDX3", neighbors=64, ef_construction=100)

    ddl = _find_ddl(cursor)
    assert "WITH DISTANCE EUCLIDEAN" in ddl
    assert "PARAMETERS (NEIGHBORS 64 EFCONSTRUCTION 100)" in ddl
    assert "ACCURACY" not in ddl


def test_create_index_cosine_succeeds() -> None:
    """create_index with COSINE issues the correct DDL (COSINE is supported)."""
    db2vs, client = _make_db2vs(DistanceStrategy.COSINE)
    cursor = client.cursor.return_value

    db2vs.create_index("VIDX_COSINE")

    ddl = _find_ddl(cursor)
    assert '"VIDX_COSINE"' in ddl
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


def test_create_index_invalid_if_exists_raises() -> None:
    """create_index raises ValueError for an unrecognised if_exists value."""
    db2vs, _ = _make_db2vs()
    with pytest.raises((ValueError, RuntimeError)):
        db2vs.create_index("BAD_IDX", if_exists="overwrite")


def test_create_index_if_exists_skip_returns_early() -> None:
    """create_index with if_exists='skip' does not issue DDL when index exists."""
    db2vs, client = _make_db2vs()
    # Simulate index already present in SYSCAT
    client.cursor.return_value.fetchone.return_value = (1,)

    db2vs.create_index("VIDX1", if_exists="skip")

    # No CREATE VECTOR INDEX should have been executed
    for c in client.cursor.return_value.execute.call_args_list:
        assert "CREATE VECTOR INDEX" not in c[0][0]


def test_create_index_if_exists_replace_drops_then_creates() -> None:
    """create_index with if_exists='replace' drops old index then creates new one."""
    db2vs, client = _make_db2vs()
    cursor = client.cursor.return_value

    # First fetchone call (existence check) → exists; subsequent → not found
    cursor.fetchone.side_effect = [(1,), None]

    db2vs.create_index("VIDX1", if_exists="replace")

    all_calls = [c[0][0] for c in cursor.execute.call_args_list]
    # DROP must appear before CREATE VECTOR INDEX
    drop_pos   = next(i for i, s in enumerate(all_calls) if "DROP INDEX" in s)
    create_pos = next(i for i, s in enumerate(all_calls) if "CREATE VECTOR INDEX" in s)
    assert drop_pos < create_pos


def test_create_index_if_exists_error_raises_when_exists() -> None:
    """create_index with if_exists='error' raises when the index already exists."""
    db2vs, client = _make_db2vs()
    client.cursor.return_value.fetchone.return_value = (1,)

    with pytest.raises((ValueError, RuntimeError)):
        db2vs.create_index("VIDX1", if_exists="error")


def test_create_index_runs_runstats_after_creation() -> None:
    """create_index calls RUNSTATS via SYSPROC.ADMIN_CMD after creating the index."""
    db2vs, client = _make_db2vs()
    cursor = client.cursor.return_value

    db2vs.create_index("VIDX1")

    all_calls = [c[0][0] for c in cursor.execute.call_args_list]
    runstats_calls = [s for s in all_calls if "RUNSTATS" in s.upper()]
    assert len(runstats_calls) >= 1
    assert "SYSPROC.ADMIN_CMD" in runstats_calls[0]


# ---------------------------------------------------------------------------
# drop_index
# ---------------------------------------------------------------------------


def test_drop_index_executes_drop_and_commit() -> None:
    """drop_index issues DROP INDEX … and COMMIT when the index exists."""
    client = MagicMock()
    cursor = client.cursor.return_value

    drop_index(client, "VIDX1")

    executed = [c[0][0] for c in cursor.execute.call_args_list]
    assert executed[0] == "DROP INDEX VIDX1"
    assert executed[1] == "COMMIT"


def test_drop_index_silently_ignores_missing_index() -> None:
    """drop_index does not raise when SQL0204N (not found) is returned."""
    client = MagicMock()
    cursor = client.cursor.return_value
    cursor.execute.side_effect = [Exception("SQL0204N index not found"), None]

    # Should not raise
    drop_index(client, "NONEXISTENT_IDX")
