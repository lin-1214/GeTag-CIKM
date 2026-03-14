from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter='__',
        extra='forbid',
        frozen=True,
        use_attribute_docstrings=True,
    )
