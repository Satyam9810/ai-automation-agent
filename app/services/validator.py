"""
Input Validation Service
------------------------
Enforces all input guards BEFORE touching the LLM.
Cheap fail-fast checks. No I/O. Pure computation.

Guards applied:
  1. Field presence (Pydantic handles this)
  2. Character length limits (instruction + document)
  3. Combined length limit (total prompt budget)
  4. Basic sanity checks (not just whitespace)
"""

from app.core.config import get_settings
from app.core.errors import ErrorCode, WorkflowError
from app.models.schemas import ProcessRequest

settings = get_settings()


def validate_request_input(request: ProcessRequest) -> None:
    """
    Raises WorkflowError if any input guard fails.
    Call this before any LLM interaction.
    """
    # ── Instruction guards ─────────────────────────────────────────────────
    if len(request.instruction) > settings.max_instruction_chars:
        raise WorkflowError(
            ErrorCode.INPUT_TOO_LARGE,
            f"Instruction exceeds maximum length of "
            f"{settings.max_instruction_chars:,} characters. "
            f"Received: {len(request.instruction):,}.",
        )

    if not request.instruction.strip():
        raise WorkflowError(
            ErrorCode.INSTRUCTION_MISSING,
            "Instruction cannot be blank or whitespace only.",
        )

    # ── Document guards ────────────────────────────────────────────────────
    if len(request.document) > settings.max_document_chars:
        raise WorkflowError(
            ErrorCode.INPUT_TOO_LARGE,
            f"Document exceeds maximum length of "
            f"{settings.max_document_chars:,} characters. "
            f"Received: {len(request.document):,}. "
            f"Please truncate or summarise the document before submitting.",
        )

    if not request.document.strip():
        raise WorkflowError(
            ErrorCode.DOCUMENT_MISSING,
            "Document cannot be blank or whitespace only.",
        )

    # ── Combined size guard ────────────────────────────────────────────────
    combined = len(request.instruction) + len(request.document)
    if combined > settings.max_combined_chars:
        raise WorkflowError(
            ErrorCode.INPUT_TOO_LARGE,
            f"Combined input length ({combined:,} chars) exceeds the service limit "
            f"of {settings.max_combined_chars:,} characters.",
        )
