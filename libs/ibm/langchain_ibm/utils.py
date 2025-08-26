import functools
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union

from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
from ibm_watsonx_ai.foundation_models.schema import BaseSchema  # type: ignore
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure  # type: ignore
from pydantic import SecretStr

logger = logging.getLogger(__name__)


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


def extract_chat_params(
    kwargs: Dict[str, Any],
    default_params: Optional[Union[BaseSchema, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
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


def check_duplicate_chat_params(params: dict, kwargs: dict) -> None:
    duplicate_keys = {k for k, v in kwargs.items() if v is not None and k in params}

    if duplicate_keys:
        raise ValueError(
            f"Duplicate parameters found in params and keyword arguments: "
            f"{list(duplicate_keys)}"
        )


def gateway_error_handler(func: Callable) -> Callable:
    """Decorator to catch ApiRequestFailure on Model Gateway calls
    and log a uniform warning."""

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
                    "Gateway. If you intend to use a watsonx.ai–hosted model, pass "
                    "'model_id' instead of 'model'."
                )
            raise

    return wrapper


def async_gateway_error_handler(func: Callable) -> Callable:
    """Async decorator to catch ApiRequestFailure on Model Gateway calls
    and log a uniform warning."""

    @functools.wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await func(self, *args, **kwargs)
        except ApiRequestFailure:
            if getattr(self, "watsonx_model_gateway", None) is not None:
                logger.warning(
                    "You are calling the Model Gateway endpoint using the 'model' "
                    "parameter. Please ensure this model is registered with the "
                    "Gateway. If you intend to use a watsonx.ai–hosted model, pass "
                    "'model_id' instead of 'model'."
                )
            raise

    return wrapper


def resolve_watsonx_credentials(
    url: SecretStr,
    apikey: SecretStr | None = None,
    token: SecretStr | None = None,
    password: SecretStr | None = None,
    username: SecretStr | None = None,
    instance_id: SecretStr | None = None,
    version: SecretStr | None = None,
    verify: bool | str | None = None,
) -> Credentials:
    check_for_attribute(url, "url", "WATSONX_URL")

    if url.get_secret_value() in APIClient.PLATFORM_URLS_MAP:
        if not token and not apikey:
            raise ValueError(
                "Did not find 'apikey' or 'token',"
                " please add an environment variable"
                " `WATSONX_APIKEY` or 'WATSONX_TOKEN' "
                "which contains it,"
                " or pass 'apikey' or 'token'"
                " as a named parameter."
            )
    else:
        if not token and not password and not apikey:
            raise ValueError(
                "Did not find 'token', 'password' or 'apikey',"
                " please add an environment variable"
                " `WATSONX_TOKEN`, 'WATSONX_PASSWORD' or 'WATSONX_APIKEY' "
                "which contains it,"
                " or pass 'token', 'password' or 'apikey'"
                " as a named parameter."
            )
        elif token:
            check_for_attribute(token, "token", "WATSONX_TOKEN")
        elif password:
            check_for_attribute(password, "password", "WATSONX_PASSWORD")
            check_for_attribute(username, "username", "WATSONX_USERNAME")
        elif apikey:
            check_for_attribute(apikey, "apikey", "WATSONX_APIKEY")
            check_for_attribute(username, "username", "WATSONX_USERNAME")

    return Credentials(
        url=url.get_secret_value() if url else None,
        api_key=apikey.get_secret_value() if apikey else None,
        token=token.get_secret_value() if token else None,
        password=password.get_secret_value() if password else None,
        username=username.get_secret_value() if username else None,
        instance_id=instance_id.get_secret_value() if instance_id else None,
        version=version.get_secret_value() if version else None,
        verify=verify,
    )
