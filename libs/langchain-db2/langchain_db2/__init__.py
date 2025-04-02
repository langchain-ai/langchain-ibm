from importlib import metadata

from langchain_db2.db2vs import DB2VS

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "DB2VS",
    "__version__",
]
