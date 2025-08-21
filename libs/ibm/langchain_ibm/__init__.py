import importlib
import warnings
from typing import Any

from langchain_ibm.chat_models import ChatWatsonx
from langchain_ibm.embeddings import WatsonxEmbeddings
from langchain_ibm.llms import WatsonxLLM
from langchain_ibm.rerank import WatsonxRerank

__all__ = [
    "WatsonxLLM",
    "WatsonxEmbeddings",
    "ChatWatsonx",
    "WatsonxRerank",
]


_module_lookup = {
    "WatsonxTool": "langchain_ibm.agent_toolkits.utility.toolkit",
    "WatsonxToolkit": "langchain_ibm.agent_toolkits.utility.toolkit",
}


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    if name in _module_lookup:
        warnings.warn(
            (
                f"Import path `from langchain_ibm import {name}` is deprecated "
                "and may be removed in future. "
                f"Use `from langchain_ibm.agent_toolkits.utility import {name}` instead."  # noqa: E501
            ),
            category=DeprecationWarning,
        )
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
