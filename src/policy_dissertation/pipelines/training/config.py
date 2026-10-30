"""Explicit, bounded Linux/CUDA compression experiment configuration."""
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class CompressionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_id: Literal['Qwen/Qwen3-14B'] = 'Qwen/Qwen3-14B'
    # Replaced by the immutable enterprise tokenizer/model revision before execution.
    model_revision: str | None = None
    prune_ratio: float = Field(default=.1, gt=0, le=.2)
    channel_multiple: int = Field(default=128, ge=1, le=1024)
    calibration_samples: int = Field(default=128, ge=1, le=2048)
    max_tokens: int = Field(default=4096, ge=32, le=32768)
    max_steps: int = Field(default=200, ge=1, le=100000)
    batch_size: int = Field(default=1, ge=1, le=16)
    gradient_accumulation: int = Field(default=16, ge=1, le=1024)
    learning_rate: float = Field(default=2e-5, gt=0, le=.001)
    lora_rank: int = Field(default=16, ge=1, le=128)
    save_steps: int = Field(default=50, ge=1, le=10000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    enterprise_repeats: int = Field(default=1, ge=1, le=100)
    minimum_free_vram_gib: float = Field(default=36, ge=1, le=1000)
    minimum_free_disk_gib: float = Field(default=120, ge=1, le=10000)
    allow_incomplete_sources: bool = False
    export_format: Literal['bitsandbytes-nf4'] = 'bitsandbytes-nf4'
