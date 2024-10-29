from typing import Any, Dict, Optional, Union

from ibm_watsonx_ai.foundation_models.schema import BaseSchema  # type: ignore
from pydantic import SecretStr


def check_for_attribute(value: SecretStr | None, key: str, env_key: str) -> None:
    if not value or not value.get_secret_value():
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )


def extract_params(
    kwargs: Dict[str, Any],
    default_params: Optional[Union[BaseSchema, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if kwargs.get("params") is not None:
        params = kwargs.pop("params")
    elif default_params is not None:
        params = default_params
    else:
        params = None

    if isinstance(params, BaseSchema):
        params = params.to_dict()

    return params or {}
