"""
Error Contract
--------------
All domain errors are typed. Each maps to a deterministic HTTP status code
and a stable `error_code` string the frontend/caller can rely on.

Never leak internal details (stack traces, raw LLM output) to the caller.
Log the raw details server-side instead.
"""

from enum import Enum


class ErrorCode(str, Enum):
    # Input validation errors → 400
    INPUT_TOO_LARGE = "input_too_large"
    INSTRUCTION_MISSING = "instruction_missing"
    DOCUMENT_MISSING = "document_missing"

    # LLM / processing errors → 502
    LLM_NON_JSON_RESPONSE = "llm_non_json_response"
    LLM_SCHEMA_INVALID = "llm_schema_invalid"
    LLM_TIMEOUT = "llm_timeout"
    LLM_API_ERROR = "llm_api_error"
    LLM_EMPTY_RESPONSE = "llm_empty_response"

    # Retry exhausted — use most recent underlying error code
    RETRIES_EXHAUSTED = "retries_exhausted"

    # Catch-all
    INTERNAL_ERROR = "internal_error"


# Maps ErrorCode → suggested HTTP status code
ERROR_STATUS_MAP: dict[ErrorCode, int] = {
    ErrorCode.INPUT_TOO_LARGE:        400,
    ErrorCode.INSTRUCTION_MISSING:    400,
    ErrorCode.DOCUMENT_MISSING:       400,
    ErrorCode.LLM_NON_JSON_RESPONSE:  502,
    ErrorCode.LLM_SCHEMA_INVALID:     502,
    ErrorCode.LLM_TIMEOUT:            504,
    ErrorCode.LLM_API_ERROR:          502,
    ErrorCode.LLM_EMPTY_RESPONSE:     502,
    ErrorCode.RETRIES_EXHAUSTED:      502,
    ErrorCode.INTERNAL_ERROR:         500,
}


class WorkflowError(Exception):
    """Base exception for all domain errors in this service."""

    def __init__(self, code: ErrorCode, detail: str = "", *, internal: str = ""):
        self.code = code
        self.detail = detail or code.value.replace("_", " ").capitalize()
        self.internal = internal  # logged server-side only, never returned to caller
        super().__init__(self.detail)

    @property
    def status_code(self) -> int:
        return ERROR_STATUS_MAP.get(self.code, 500)

    def to_response(self) -> dict:
        return {
            "error": self.code.value,
            "detail": self.detail,
        }
