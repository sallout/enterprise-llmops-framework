"""Code-specific contracts extending Pipeline 1's conversational record metadata."""
from typing import Literal
from pydantic import Field, StrictBool
from policy_dissertation.pipelines.policy_transformation.data_schemas import StrictModel, ConversationRecord

VERSION = 'repository-data-v1'
TASK_TYPES = ('implementation', 'explanation', 'debugging', 'unit_tests', 'refactoring', 'architecture', 'secure_coding')
TaskType = Literal['implementation', 'explanation', 'debugging', 'unit_tests', 'refactoring', 'architecture', 'secure_coding']
LanguageName = Literal['python', 'kotlin', 'typescript']


class CodeUnit(StrictModel):
    unit_id: str
    repository_id: str
    repository_name: str
    repository_path: str
    commit: str | None
    snapshot_hash: str
    file_path: str
    file_hash: str
    symbol: str
    kind: str
    language: LanguageName
    start_line: int
    end_line: int
    code: str
    imports: list[str]
    syntax_valid: bool
    is_test: bool
    token_count: int
    related_tests: list[dict] = Field(default_factory=list)
    dependencies: list[dict] = Field(default_factory=list)
    local_context: list[dict] = Field(default_factory=list)
    licenses: list[dict] = Field(default_factory=list)


class CodeCandidate(StrictModel):
    task_type: TaskType
    instruction: str = Field(min_length=20, max_length=10000)
    response: str = Field(min_length=20, max_length=20000)
    source_unit_id: str
    source_quotes: list[str] = Field(min_length=1, max_length=4)
    language: LanguageName
    solution: str | None = Field(max_length=20000)
    tests: str | None = Field(max_length=20000)


class CodeEnvelope(StrictModel):
    examples: list[dict] = Field(min_length=1, max_length=1)


class CodeVerdict(StrictModel):
    grounded: StrictBool
    self_contained: StrictBool
    technically_sound: StrictBool
    tests_meaningful: StrictBool
    reason: str = Field(min_length=1, max_length=800)


class Repair(StrictModel):
    solution: str = Field(min_length=1, max_length=20000)
    correction_summary: str = Field(min_length=1, max_length=1500)


class ExecutionResult(StrictModel):
    status: Literal['verified', 'unverified', 'failed']
    executor: str
    reason: str
    command: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    tests_run: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    expected_failures: int = 0
    unexpected_successes: int = 0
    feedback: str = ''
    image_id: str | None = None
    duration_seconds: float = 0


class CodeTrainingRecord(ConversationRecord):
    schema_version: Literal['repository-data-v1'] = VERSION
