"""Utility helpers for langchain-ibm."""

import functools
import logging
import os
import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast
from urllib.parse import urlparse

from ibm_watsonx_ai import APIClient, Credentials  # type: ignore[import-untyped]
from ibm_watsonx_ai.foundation_models.schema import (  # type: ignore[import-untyped]
    BaseSchema,
)
from ibm_watsonx_ai.wml_client_error import (  # type: ignore[import-untyped]
    ApiRequestFailure,
)
from pydantic import SecretStr

logger = logging.getLogger(__name__)


def check_for_attribute(value: SecretStr | None, key: str, env_key: str) -> None:
    """Check for attribute."""
    if not value or not value.get_secret_value():
        error_msg = (
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )
        raise ValueError(error_msg)


def extract_params(
    kwargs: dict[str, Any],
    default_params: BaseSchema | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract params."""
    if kwargs.get("params") is not None:
        params = kwargs.pop("params")
    elif default_params is not None:
        params = default_params
    else:
        params = None

    if isinstance(params, BaseSchema):
        params = params.to_dict()

    return params or {}


def extract_chat_params(
    kwargs: dict[str, Any],
    default_params: BaseSchema | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract chat params."""
    if kwargs.get("params") is not None:
        params = kwargs.pop("params")
        check_duplicate_chat_params(params, kwargs)
    elif default_params is not None:
        params = deepcopy(default_params)
    else:
        params = None

    if isinstance(params, BaseSchema):
        params = params.to_dict()

    return params or {}


def check_duplicate_chat_params(params: dict[str, Any], kwargs: dict[str, Any]) -> None:
    """Check duplicate chat params."""
    duplicate_keys = {k for k, v in kwargs.items() if v is not None and k in params}

    if duplicate_keys:
        error_msg = (
            f"Duplicate parameters found in params and keyword arguments: "
            f"{list(duplicate_keys)}"
        )
        raise ValueError(error_msg)


def gateway_error_handler(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to catch ApiRequestFailure on Model Gateway calls.

    Logs a uniform warning when the model is not properly registered.
    """

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
        except ApiRequestFailure:
            if any(
                hasattr(self, attr)
                for attr in ("watsonx_model_gateway", "watsonx_embed_gateway")
            ):
                logger.warning(
                    "You are calling the Model Gateway endpoint using the 'model' "
                    "parameter. Please ensure this model is registered with the "
                    "Gateway. If you intend to use a watsonx.ai-hosted model, pass "
                    "'model_id' instead of 'model'.",
                )
            raise

    return wrapper


def async_gateway_error_handler(func: Callable[..., Any]) -> Callable[..., Any]:
    """Async decorator to catch ApiRequestFailure on Model Gateway calls.

    Log a uniform warning when the Model Gateway is misused.
    """

    @functools.wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await func(self, *args, **kwargs)
        except ApiRequestFailure:
            if getattr(self, "watsonx_model_gateway", None) is not None:
                logger.warning(
                    "You are calling the Model Gateway endpoint using the 'model' "
                    "parameter. Please ensure this model is registered with the "
                    "Gateway. If you intend to use a watsonx.ai-hosted model, pass "
                    "'model_id' instead of 'model'.",
                )
            raise

    return wrapper


def resolve_watsonx_credentials(
    url: SecretStr,
    *,
    api_key: SecretStr | None = None,
    token: SecretStr | None = None,
    password: SecretStr | None = None,
    username: SecretStr | None = None,
    instance_id: SecretStr | None = None,
    version: SecretStr | None = None,
    verify: bool | str | None = None,
) -> Credentials:
    """Resolve watsonx credentials."""
    check_for_attribute(url, "url", "WATSONX_URL")

    raw_url = url.get_secret_value()
    parsed_url = urlparse(raw_url)
    clean_url = f"{parsed_url.scheme}://{parsed_url.hostname}"

    if clean_url in APIClient.PLATFORM_URLS_MAP:
        if not token and not api_key:
            error_msg = (
                "Did not find 'api_key' or 'token',"
                " please add an environment variable"
                " `WATSONX_API_KEY` or 'WATSONX_TOKEN' "
                "which contains it,"
                " or pass 'api_key' or 'token'"
                " as a named parameter."
            )
            raise ValueError(error_msg)
    elif not token and not password and not api_key:
        error_msg = (
            "Did not find 'token', 'password' or 'api_key',"
            " please add an environment variable"
            " `WATSONX_TOKEN`, 'WATSONX_PASSWORD' or 'WATSONX_API_KEY' "
            "which contains it,"
            " or pass 'token', 'password' or 'api_key'"
            " as a named parameter."
        )
        raise ValueError(error_msg)
    elif token:
        check_for_attribute(token, "token", "WATSONX_TOKEN")
    elif password:
        check_for_attribute(password, "password", "WATSONX_PASSWORD")
        check_for_attribute(username, "username", "WATSONX_USERNAME")
    elif api_key:
        check_for_attribute(api_key, "api_key", "WATSONX_API_KEY")
        check_for_attribute(username, "username", "WATSONX_USERNAME")

    return Credentials(
        url=url.get_secret_value() if url else None,
        api_key=api_key.get_secret_value() if api_key else None,
        token=token.get_secret_value() if token else None,
        password=password.get_secret_value() if password else None,
        username=username.get_secret_value() if username else None,
        instance_id=instance_id.get_secret_value() if instance_id else None,
        version=version.get_secret_value() if version else None,
        verify=verify,
    )


def secret_from_env_multi(
    names_priority: list[str], deprecated: set[str] | None = None
) -> Callable[[], SecretStr | None]:
    """Return default factory that yields a SecretStr from the first non-empty env var.

    The factory:
    - Warns if multiple environment variables are set
        (uses the first in `names_priority`).
    - Warns if the chosen environment variable is listed in `deprecated`.
    """
    deprecated = deprecated or set()

    def _factory() -> SecretStr | None:
        present = [(n, os.getenv(n)) for n in names_priority]
        present = [(n, v) for n, v in present if v not in (None, "")]
        if not present:
            return None

        chosen_name, value = present[0]
        if len(present) > 1:
            warnings.warn(
                f"Multiple API key env vars are set {[n for n, _ in present]}; "
                f"using '{chosen_name}'.",
                UserWarning,
                stacklevel=2,
            )
        if chosen_name in deprecated:
            warnings.warn(
                f"Environment variable '{chosen_name}' is deprecated; "
                f"use '{names_priority[0]}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return SecretStr(cast("str", value))

    return _factory


def normalize_api_key(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize deprecated 'apikey' to 'api_key'.

    - If only 'apikey' is provided, convert it to 'api_key' with a DeprecationWarning.
    - If both are provided, 'api_key' takes precedence with a UserWarning.
    """
    has_new = "api_key" in data
    has_old = "apikey" in data

    if has_old and not has_new:
        warnings.warn(
            "'apikey' parameter is deprecated; use 'api_key' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        data = {**data, "api_key": data.pop("apikey")}
    elif has_old and has_new:
        warnings.warn(
            "Both 'api_key' and deprecated 'apikey' were provided; "
            "'api_key' takes precedence.",
            UserWarning,
            stacklevel=2,
        )
        data = {**data}
        data.pop("apikey", None)

    return data
