from pydantic import SecretStr


def check_for_attribute(key: SecretStr | None, env_key: str) -> None:
    if not (key.get_secret_value() if key else None):
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )
