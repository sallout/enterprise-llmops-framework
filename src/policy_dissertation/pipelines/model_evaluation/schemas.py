"""Explicit evaluation contract, separate from training records."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Case(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str
    suite: Literal['humaneval', 'mbpp', 'enterprise']
    messages: list[dict]
    reference: str
    tests: str = ''
    source: dict


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_new_tokens: int = Field(default=1024, ge=16, le=8192)
    max_input_tokens: int = Field(default=4096, ge=32, le=32768)
    seed: int = Field(default=42, ge=0, le=2147483647)
    executor: Literal['docker', 'disabled'] = 'docker'
    python_image: str = 'python:3.12-slim'
    timeout_seconds: int = Field(default=15, ge=1, le=120)
    thinking: Literal[False] = False
