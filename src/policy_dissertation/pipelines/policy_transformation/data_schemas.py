"""Canonical conversational records and auditable generation/validation contracts."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool

DATA_VERSION = 'pipeline1-extractive-v1'


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Candidate(StrictModel):
    instruction: str = Field(min_length=12, max_length=1500)
    response: str = Field(min_length=20, max_length=20000)
    document_id: str
    chunk_id: str
    passage_id: str


class GenerationEnvelope(StrictModel):
    # Individual objects are validated separately so valid siblings can survive.
    examples: list[dict] = Field(min_length=1, max_length=3)


class GroundingVerdict(StrictModel):
    grounded: StrictBool
    answers_question: StrictBool
    preserves_conditions: StrictBool
    scope_correct: StrictBool
    reason: str = Field(min_length=1, max_length=600)


class SystemMessage(StrictModel):
    role: Literal['system'] = 'system'
    content: str = Field(min_length=1)


class UserMessage(StrictModel):
    role: Literal['user'] = 'user'
    content: str = Field(min_length=12)


class AssistantMessage(StrictModel):
    role: Literal['assistant'] = 'assistant'
    content: str = Field(min_length=20)


class ConversationArtifact(StrictModel):
    """Common export fields; pipelines specialize messages and typed metadata."""
    schema_version: str
    example_id: str
    messages: list[dict]
    metadata: dict


class ConversationRecord(ConversationArtifact):
    schema_version: str
    example_id: str
    messages: tuple[SystemMessage, UserMessage, AssistantMessage]
    metadata: dict


class TrainingRecord(ConversationRecord):
    schema_version: Literal['pipeline1-extractive-v1'] = DATA_VERSION


class ValidationOutcome(StrictModel):
    chunk_id: str
    generated_count: int
    complete: bool
    accepted: list[TrainingRecord]
    rejected: list[dict]
    errors: list[str]
