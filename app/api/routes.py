"""
API Routes
----------
One endpoint. Thin controller — no business logic lives here.
It orchestrates: validate → call LLM → return result.

Error handling converts all WorkflowError to clean JSON responses.
FastAPI's built-in RequestValidationError (from Pydantic) is also caught
and reshaped to match our error contract.
"""

import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.errors import WorkflowError
from app.core.logging import get_logger, log_request_event
from app.models.schemas import ProcessRequest, ProcessResponse
from app.services.gemini_client import call_gemini
from app.services.validator import validate_request_input

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/process",
    response_model=ProcessResponse,
    responses={
        400: {"description": "Invalid input"},
        502: {"description": "AI service error"},
        504: {"description": "AI service timeout"},
    },
    summary="Convert an instruction + document into a structured workflow plan.",
)
async def process_document(
    body: ProcessRequest,
    request: Request,
) -> JSONResponse:
    request_id = str(uuid.uuid4())
    t_start = time.monotonic()

    log_request_event(
        logger,
        request_id,
        "request_received",
        instruction_chars=len(body.instruction),
        document_chars=len(body.document),
        ip=request.client.host if request.client else "unknown",
    )

    # ── 1. Input validation ────────────────────────────────────────────────
    try:
        validate_request_input(body)
    except WorkflowError as exc:
        latency_ms = int((time.monotonic() - t_start) * 1000)
        log_request_event(
            logger, request_id, "validation_failed",
            error_code=exc.code.value,
            latency_ms=latency_ms,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={**exc.to_response(), "request_id": request_id},
        )

    # ── 2. LLM call ────────────────────────────────────────────────────────
    try:
        result, meta = await call_gemini(body, request_id)
    except WorkflowError as exc:
        latency_ms = int((time.monotonic() - t_start) * 1000)
        log_request_event(
            logger, request_id, "request_failed",
            error_code=exc.code.value,
            latency_ms=latency_ms,
            internal_detail=exc.internal,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={**exc.to_response(), "request_id": request_id},
        )
    except Exception as exc:
        latency_ms = int((time.monotonic() - t_start) * 1000)
        log_request_event(
            logger, request_id, "request_unhandled_error",
            error=str(exc),
            latency_ms=latency_ms,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": "An unexpected error occurred.",
                "request_id": request_id,
            },
        )

    # ── 3. Success ─────────────────────────────────────────────────────────
    latency_ms = int((time.monotonic() - t_start) * 1000)
    meta_clean = {k: v for k, v in meta.items() if k != "latency_ms"}
    log_request_event(
        logger, request_id, "request_success",
        latency_ms=latency_ms,
        risks_count=len(result.risks),
        action_items_count=len(result.action_items),
        **meta_clean,
    )

    return JSONResponse(
        status_code=200,
        content={
            "request_id": request_id,
            "result": result.model_dump(),
        },
    )