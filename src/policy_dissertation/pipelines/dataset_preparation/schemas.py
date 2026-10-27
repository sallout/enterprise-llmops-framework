"""NVIDIA-inspired messages plus separately auditable verification and provenance."""
from typing import Literal
from pydantic import Field, model_validator
from policy_dissertation.pipelines.policy_transformation.data_schemas import StrictModel
from policy_dissertation.pipelines.external_datasets.schemas import ExternalMessage, ExternalTrainingRecord

VERSION = 'unified-qwen-sft-v1'


class Verification(StrictModel):
    status: Literal['not_applicable', 'upstream_local_pass', 'externally_reported', 'unverified']
    executed_in_pipeline4: Literal[False] = False
    unit_tests: str | list | dict | None = None
    evidence: dict = Field(default_factory=dict)


class UnifiedRecord(StrictModel):
    schema_version: Literal['unified-qwen-sft-v1'] = VERSION
    id: str = Field(pattern=r'^[0-9a-f]{64}$')
    messages: list[ExternalMessage] = Field(min_length=2)
    domain: str = Field(min_length=1)
    source: str = Field(min_length=1)
    verification: Verification
    metadata: dict
    tools: list[dict] | None = None

    @model_validator(mode='after')
    def valid_protocol(self):
        # Reuse the existing role/order/call-result validator; it only accesses .messages.
        ExternalTrainingRecord.conversation(self)
        return self

    def export_dict(self):
        data = self.model_dump()
        data['messages'] = [m.model_dump(exclude_unset=True) for m in self.messages]
        if self.tools is None:
            data.pop('tools')
        return data
