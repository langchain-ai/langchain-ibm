from pydantic import SecretStr


def check_for_attribute(value: SecretStr | None, key: str, env_key: str) -> None:
    if not value or not value.get_secret_value():
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )
