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


def _load_env() -> None:
    dotenv_path = Path(PROJECT_DIR) / "tests" / "integration_tests" / ".env"
    if Path(dotenv_path).exists():
        load_dotenv(dotenv_path)


_load_env()


def _build_dsn(user: str, password: str) -> str:
    """Build a Db2 DSN string from environment variables.

    SSL is included only when DB2_SSL=true in the environment.
    """
    db2_name = os.environ.get("DB2_NAME", "")
    db2_host = os.environ.get("DB2_HOST", "")
    db2_port = os.environ.get("DB2_PORT", "50000")
    use_ssl = os.environ.get("DB2_SSL", "false").lower() == "true"

    dsn = (
        f"DATABASE={db2_name};hostname={db2_host};port={db2_port};"
        f"uid={user};pwd={password};"
    )
    if use_ssl:
        dsn += "SECURITY=SSL;"
    return dsn


@pytest.fixture(scope="session")
def ibm_db_dbi_connection() -> Generator[ibm_db_dbi.Connection, None, None]:
    user = os.environ.get("DB2_USER", "")
    password = os.environ.get("DB2_PASSWORD", "")
    conn = ibm_db_dbi.connect(_build_dsn(user, password), "", "")
    try:
        yield conn
    finally:
        with contextlib.suppress(Exception):
            conn.commit()
        with contextlib.suppress(Exception):
            conn.close()


@pytest.fixture(scope="session")
def limited_privilege_connection() -> Generator[ibm_db_dbi.Connection, None, None]:
    """Connection for a non-DBADM user (DB2_LIMITED_USER / DB2_LIMITED_PASSWORD).

    Used to verify that CREATE VECTOR INDEX does not require DBADM authority
    in Db2 12.1.5.0 (Mod Pack 5) and later.
    """
    user = os.environ.get("DB2_LIMITED_USER", "")
    password = os.environ.get("DB2_LIMITED_PASSWORD", "")
    if not user or not password:
        pytest.skip("DB2_LIMITED_USER / DB2_LIMITED_PASSWORD not configured")
    conn = ibm_db_dbi.connect(_build_dsn(user, password), "", "")
    try:
        yield conn
    finally:
        with contextlib.suppress(Exception):
            conn.commit()
        with contextlib.suppress(Exception):
            conn.close()


@pytest.fixture(scope="session")
def hf_embeddings() -> HuggingFaceEmbeddings:
    """Load the HuggingFace embedding model once per test session.

    Avoids re-downloading or re-initialising for every test.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
