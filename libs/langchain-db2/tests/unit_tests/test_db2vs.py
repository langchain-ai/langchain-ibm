from unittest.mock import MagicMock

from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from langchain_db2.db2vs import DB2VS


def test_init() -> None:
    """Test that the DB2VS class can be initialized."""
    client = MagicMock()
    embedding = DeterministicFakeEmbedding(size=100)
    table_name = "foo"
    db2vs = DB2VS(client, embedding, table_name)
    assert db2vs is not None
    assert isinstance(db2vs, DB2VS)
    assert len(client.mock_calls) == 3
