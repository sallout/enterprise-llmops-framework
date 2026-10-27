"""Multi-turn extension of the project's canonical conversational artifact."""
import json
from typing import Literal
from pydantic import Field, model_validator
from policy_dissertation.pipelines.policy_transformation.data_schemas import StrictModel, ConversationArtifact

VERSION = 'external-datasets-v1'


class ToolFunction(StrictModel):
    name: str = Field(min_length=1)
    arguments: str | dict

    @model_validator(mode='after')
    def valid_arguments(self):
        def reject_constant(value):
            raise ValueError('Nonfinite tool argument')
        if isinstance(self.arguments, str) and not isinstance(json.loads(self.arguments, parse_constant=reject_constant), dict):
            raise ValueError('Tool arguments must encode an object')
        return self


class ToolCall(StrictModel):
    id: str = Field(min_length=1)
    type: Literal['function']
    function: ToolFunction


class ExternalMessage(StrictModel):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: str | None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class SourceMetadata(StrictModel):
    dataset: str
    config: str
    split: str
    revision: str = Field(min_length=1)
    record_id: str
    record_id_kind: Literal['upstream', 'stream_ordinal']
    stream_ordinal: int = Field(ge=0)
    content_hash: str
    source_url: str
    license: str
    license_url: str
    source_fields: dict = Field(default_factory=dict)
    source_message_metadata: dict = Field(default_factory=dict)
    transformations: list[dict] = Field(default_factory=list)
    token_count: int = Field(ge=0)
    token_method: Literal['utf8_serialized_messages_bytes_v1'] = 'utf8_serialized_messages_bytes_v1'
    locally_executed: Literal[False] = False
    semantic_status: Literal['structural_checks_only'] = 'structural_checks_only'
    external_test_results_status: Literal['not_locally_verified', 'not_provided']


class ExternalTrainingRecord(ConversationArtifact):
    schema_version: Literal['external-datasets-v1'] = VERSION
    messages: list[ExternalMessage] = Field(min_length=2)
    metadata: SourceMetadata

    @model_validator(mode='after')
    def conversation(self):
        seen_user = False
        pending = set()
        used_ids = set()
        previous = None
        for m in self.messages:
            if m.role == 'system':
                if seen_user or m.content is None:
                    raise ValueError('System messages must precede dialogue')
            elif m.role == 'user':
                if pending or not m.content or not m.content.strip():
                    raise ValueError('Empty user or unanswered tool calls')
                seen_user = True
            elif m.role == 'assistant':
                if not seen_user or pending or previous not in ('user', 'tool'):
                    raise ValueError('Assistant must follow user or completed tools')
                if m.tool_calls:
                    for call in m.tool_calls:
                        if call.id in used_ids:
                            raise ValueError('Duplicate tool call identifier')
                        pending.add(call.id)
                        used_ids.add(call.id)
                elif not m.content or not m.content.strip():
                    raise ValueError('Assistant answer is empty')
            elif m.role == 'tool':
                if m.tool_call_id not in pending or not m.content or not m.content.strip():
                    raise ValueError('Unmatched or empty tool result')
                pending.remove(m.tool_call_id)
            if m.role != 'assistant' and m.tool_calls:
                raise ValueError('Only assistants call tools')
            if m.role != 'tool' and m.tool_call_id is not None:
                raise ValueError('Only tool results reference calls')
            previous = m.role
        if pending or self.messages[-1].role != 'assistant':
            raise ValueError('Conversation must end in a completed assistant answer')
        return self
