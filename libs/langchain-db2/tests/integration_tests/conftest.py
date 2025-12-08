import contextlib
import os
from collections.abc import Generator
from pathlib import Path

import ibm_db_dbi  # type: ignore[import-untyped]
import pytest
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# Getting the absolute path of the current file's directory
ABS_PATH = (Path(__file__)).parent

# Getting the absolute path of the project's root directory
PROJECT_DIR = Path(ABS_PATH).parent.parent


@pytest.fixture(scope="session")
def ibm_db_dbi_connection() -> Generator[ibm_db_dbi.Connection, None, None]:
    db2_name = os.environ.get("DB2_NAME", "")
    db2_host = os.environ.get("DB2_HOST", "")
    db2_port = os.environ.get("DB2_PORT", "")
    db2_user = os.environ.get("DB2_USER", "")
    db2_password = os.environ.get("DB2_PASSWORD", "")

    dsn = (
        f"DATABASE={db2_name};hostname={db2_host};port={db2_port};uid={db2_user};pwd={db2_password};"
        f"SECURITY=SSL;"
    )
    db2_connect_user = os.environ.get("DB2_CONNECT_USER", "")
    db2_connect_password = os.environ.get("DB2_CONNECT_PASSWORD", "")

    conn = ibm_db_dbi.connect(dsn, db2_connect_user, db2_connect_password)
    try:
        yield conn
    finally:
        # Best-effort cleanup
        with contextlib.suppress(Exception):
            conn.commit()
        with contextlib.suppress(Exception):
            conn.close()


@pytest.fixture(scope="session")
def hf_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace embedding model once per test session.
    This avoids re-downloading / re-initializing for every test.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def _load_env() -> None:
    dotenv_path = Path(PROJECT_DIR) / "tests" / "integration_tests" / ".env"
    if Path(dotenv_path).exists():
        load_dotenv(dotenv_path)


_load_env()
