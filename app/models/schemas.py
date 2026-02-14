"""
Pydantic Models — Request / Response / LLM Output Schema
---------------------------------------------------------
These models serve triple duty:
  1. FastAPI request body validation (ProcessRequest)
  2. LLM output schema enforcement (WorkflowResult)
  3. API response shaping (ProcessResponse / ErrorResponse)

Keep models flat and explicit. No Optional fields that let the LLM
silently skip critical data — use defaults with clear sentinel values.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


# ── Request ───────────────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    instruction: str = Field(
        ...,
        min_length=10,
        description="Natural language instruction describing what to do with the document.",
    )
    document: str = Field(
        ...,
        min_length=20,
        description="The unstructured document text to be analysed.",
    )

    @field_validator("instruction")
    @classmethod
    def strip_instruction(cls, v: str) -> str:
        return v.strip()

    @field_validator("document")
    @classmethod
    def strip_document(cls, v: str) -> str:
        return v.strip()


# ── LLM Output Sub-models ─────────────────────────────────────────────────────

class Risk(BaseModel):
    description: str = Field(..., min_length=5)
    priority: Literal["high", "medium", "low"]


class ActionItem(BaseModel):
    task: str = Field(..., min_length=5)
    owner: str = Field(default="Not specified")
    deadline: str = Field(default="Not specified")


# ── LLM Output Root ───────────────────────────────────────────────────────────

class WorkflowResult(BaseModel):
    """
    This is the exact schema we demand from Gemini.
    Pydantic will reject any response that doesn't match.
    Extra fields from the LLM are silently stripped via model_config.
    """

    model_config = {"extra": "ignore"}  # strip hallucinated extra fields

    summary: str = Field(..., min_length=10)
    risks: list[Risk] = Field(..., description="May be empty list if no risks found.")
    action_items: list[ActionItem] = Field(
        ..., description="May be empty list if no action items found."
    )


# ── API Responses ─────────────────────────────────────────────────────────────

class ProcessResponse(BaseModel):
    request_id: str
    result: WorkflowResult


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
