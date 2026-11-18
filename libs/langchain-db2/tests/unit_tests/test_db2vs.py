from unittest.mock import MagicMock

import pytest
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from langchain_db2.db2vs import DB2VS, Db2DistanceStrategy, _quote_ident, drop_index


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
    assert _quote_ident("myindex") == '"MYINDEX"'


def test_quote_ident_escapes_internal_double_quotes() -> None:
    assert _quote_ident('my"idx') == '"MY""IDX"'


def test_quote_ident_empty_raises() -> None:
    with pytest.raises(ValueError):
        _quote_ident("")


def test_quote_ident_whitespace_only_raises() -> None:
    with pytest.raises(ValueError):
        _quote_ident("   ")


# ---------------------------------------------------------------------------
# drop_index
# ---------------------------------------------------------------------------

def test_drop_index_executes_drop_and_commit() -> None:
    client = MagicMock()
    drop_index(client, "VIDX1")
    cursor = client.cursor.return_value.__enter__.return_value
    client.cursor.assert_called_once()
    cursor = client.cursor.return_value
    execute_calls = cursor.execute.call_args_list
    assert any("DROP INDEX" in str(c) for c in execute_calls)


def test_drop_index_silently_ignores_missing_index() -> None:
    client = MagicMock()
    cursor = client.cursor.return_value
    cursor.execute.side_effect = [Exception("SQL0204N index not found"), None]
    drop_index(client, "NONEXISTENT")
    # no exception raised


# ---------------------------------------------------------------------------
# create_index
# ---------------------------------------------------------------------------

def _make_db2vs(distance: Db2DistanceStrategy = Db2DistanceStrategy.EUCLIDEAN_DISTANCE) -> tuple[DB2VS, MagicMock]:
    client = MagicMock()
    embedding = DeterministicFakeEmbedding(size=4)
    db2vs = DB2VS(embedding, "TEST_TABLE", client, distance_strategy=distance)
    client.reset_mock()
    return db2vs, client


def test_create_index_default_diskann() -> None:
    db2vs, client = _make_db2vs()
    cursor = client.cursor.return_value
    cursor.fetchone.return_value = None  # index does not exist
    db2vs.create_index("VIDX1")
    calls = [str(c) for c in cursor.execute.call_args_list]
    assert any("CREATE VECTOR INDEX" in c and "EUCLIDEAN" in c for c in calls)


def test_create_index_with_parallel() -> None:
    db2vs, client = _make_db2vs()
    cursor = client.cursor.return_value
    cursor.fetchone.return_value = None
    db2vs.create_index("VIDX1", parallel=8)
    calls = [str(c) for c in cursor.execute.call_args_list]
    assert any("BUILD_PARALLELISM 8" in c for c in calls)


def test_create_index_with_tuning_params() -> None:
    db2vs, client = _make_db2vs()
    cursor = client.cursor.return_value
    cursor.fetchone.return_value = None
    db2vs.create_index("VIDX1", neighbors=64, ef_construction=100)
    calls = [str(c) for c in cursor.execute.call_args_list]
    assert any("MAX_NODE_DEGREE 64" in c for c in calls)
    assert any("BUILD_LIST_SIZE 100" in c for c in calls)


def test_create_index_cosine_succeeds() -> None:
    db2vs, client = _make_db2vs(Db2DistanceStrategy.COSINE)
    cursor = client.cursor.return_value
    cursor.fetchone.return_value = None
    db2vs.create_index("VIDX_COS")
    calls = [str(c) for c in cursor.execute.call_args_list]
    assert any("COSINE" in c for c in calls)


def test_create_index_dot_product_raises() -> None:
    db2vs, client = _make_db2vs(Db2DistanceStrategy.DOT_PRODUCT)
    with pytest.raises(ValueError, match="cannot be used"):
        db2vs.create_index("VIDX1")


def test_create_index_hamming_raises() -> None:
    db2vs, client = _make_db2vs(Db2DistanceStrategy.HAMMING)
    with pytest.raises(ValueError, match="cannot be used"):
        db2vs.create_index("VIDX1")


def test_create_index_manhattan_raises() -> None:
    db2vs, client = _make_db2vs(Db2DistanceStrategy.MANHATTAN)
    with pytest.raises(ValueError, match="cannot be used"):
        db2vs.create_index("VIDX1")


def test_create_index_max_inner_product_succeeds() -> None:
    db2vs, client = _make_db2vs(Db2DistanceStrategy.MAX_INNER_PRODUCT)
    cursor = client.cursor.return_value
    cursor.fetchone.return_value = None
    db2vs.create_index("VIDX_MIP")
    calls = [str(c) for c in cursor.execute.call_args_list]
    assert any("EUCLIDEAN_SQUARED" in c for c in calls)


def test_create_index_partial_power_params_raises() -> None:
    db2vs, client = _make_db2vs()
    with pytest.raises(ValueError, match="together"):
        db2vs.create_index("VIDX1", neighbors=64)


def test_create_index_invalid_if_exists_raises() -> None:
    db2vs, client = _make_db2vs()
    with pytest.raises(ValueError, match="if_exists"):
        db2vs.create_index("VIDX1", if_exists="overwrite")


def test_create_index_if_exists_skip_returns_early() -> None:
    db2vs, client = _make_db2vs()
    cursor = client.cursor.return_value
    cursor.fetchone.return_value = (1,)  # index exists
    db2vs.create_index("VIDX1", if_exists="skip")
    calls = [str(c) for c in cursor.execute.call_args_list]
    assert not any("CREATE VECTOR INDEX" in c for c in calls)


def test_create_index_if_exists_replace_drops_then_creates() -> None:
    db2vs, client = _make_db2vs()
    cursor = client.cursor.return_value
    cursor.fetchone.return_value = (1,)  # index exists
    db2vs.create_index("VIDX1", if_exists="replace")
    calls = [str(c) for c in cursor.execute.call_args_list]
    assert any("DROP INDEX" in c for c in calls)
    assert any("CREATE VECTOR INDEX" in c for c in calls)


def test_create_index_if_exists_error_raises_when_exists() -> None:
    db2vs, client = _make_db2vs()
    cursor = client.cursor.return_value
    cursor.fetchone.return_value = (1,)  # index exists
    with pytest.raises(ValueError, match="already exists"):
        db2vs.create_index("VIDX1", if_exists="error")
