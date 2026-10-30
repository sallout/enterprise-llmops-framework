"""Validated local configuration; loading does not contact external services."""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseModel):
    """Names match the existing MODEL__* environment variables."""

    base_url: str | None = None
    api_key: SecretStr | None = Field(default=None, repr=False)
    model_name: str | None = None


class Pipeline1Settings(BaseModel):
    """Generation limits are independent of the source chunk byte-token budget."""
    response_max_tokens: int = Field(default=4096, ge=256, le=32768)
    max_attempts: int = Field(default=3, ge=1, le=5)
    request_timeout: float = Field(default=60, gt=0, le=600)


class Pipeline2Settings(Pipeline1Settings):
    response_max_tokens: int = Field(default=8192, ge=256, le=32768)
    request_timeout: float = Field(default=180, gt=0, le=600)
    max_file_bytes: int = Field(default=250000, ge=1024, le=2000000)
    max_corrections: int = Field(default=2, ge=0, le=3)
    execution_mode: Literal["auto", "disabled"] = "auto"
    python_image: str = "python:3.12-slim"
    sandbox_timeout: float = Field(default=15, gt=0, le=60)
    max_feedback_bytes: int = Field(default=12000, ge=1000, le=64000)


class Pipeline3Settings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reasoning_policy: Literal['reject', 'strip_marked'] = 'strip_marked'
    max_tokens: int = Field(default=32768, ge=1, le=1000000)
    min_assistant_chars: int = Field(default=1, ge=1)
    max_record_bytes: int = Field(default=1048576, ge=1024, le=16777216)
    shuffle_buffer: int = Field(default=16, ge=1, le=128)
    scan_multiplier: int = Field(default=10, ge=1, le=100)
    max_attempts: int = Field(default=3, ge=1, le=5)
    request_timeout: float = Field(default=30, gt=0, le=120)
    unsupported_reasoning_markers: list[str] = Field(default_factory=lambda: ['<analysis>', '</analysis>'])
    forbidden_tokens: list[str] = Field(default_factory=lambda: [
        '<|im_start|>', '<|im_end|>', '<|endoftext|>', '<|start_header_id|>',
        '<|end_header_id|>', '<|eot_id|>', '<|analysis|>', '<|begin_of_text|>'])


class Pipeline4Settings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    tokenizer_model: Literal['Qwen/Qwen3-14B'] = 'Qwen/Qwen3-14B'
    tokenizer_revision: str = 'main'
    local_files_only: bool = False
    max_attempts: int = Field(default=3, ge=1, le=5)
    max_line_bytes: int = Field(default=4194304, ge=1024, le=16777216)
    near_threshold: float = Field(default=0.9, ge=0.5, le=1)
    near_remove_threshold: float = Field(default=0.98, ge=0.5, le=1)
    reject_secrets: bool = True
    reject_sensitive_identifiers: bool = True
    allow_marked_reasoning: bool = False
    mix_ratios: dict[str, float] = Field(default_factory=lambda: {'pipeline1': .4, 'pipeline2': .3, 'pipeline3': .3})
    split_ratios: dict[str, float] = Field(default_factory=lambda: {'train': .9, 'validation': .05, 'test': .05})


class InterfaceSettings(BaseModel):
    title: str = Field(default='Bridging the Cusem', min_length=1, max_length=120)
    author: str = Field(default='Mohammed Sallout', min_length=1, max_length=100)
    linkedin_url: HttpUrl = Field(default='https://www.linkedin.com/in/msallout/', validate_default=True)


class Settings(BaseSettings):
    """Environment overrides .env; relative paths use the working directory."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_nested_delimiter="__", extra="ignore",
    )
    hf_token: SecretStr | None = Field(default=None, repr=False)
    ui: InterfaceSettings = Field(default_factory=InterfaceSettings)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    policy_dir: Path = Path("dumy_policies")
    parsed_output_dir: Path = Path("artifacts/pipeline1/parsed")
    chunk_output_dir: Path = Path("artifacts/pipeline1/chunks")
    pipeline1: Pipeline1Settings = Field(default_factory=Pipeline1Settings)
    pipeline2: Pipeline2Settings = Field(default_factory=Pipeline2Settings)
    pipeline3: Pipeline3Settings = Field(default_factory=Pipeline3Settings)
    pipeline4: Pipeline4Settings = Field(default_factory=Pipeline4Settings)
    model: ModelSettings = Field(default_factory=ModelSettings)
