"""
Gemini LLM Client
-----------------
Handles:
  - Async HTTP call to Gemini REST API (no SDK dependency → lighter cold start)
  - Strict JSON extraction from response
  - Retry logic (max 2 retries, exponential-ish backoff)
  - Timeout enforcement
  - Token usage extraction for observability
  - All LLM-level errors mapped to WorkflowError

Why REST over SDK?
  The google-generativeai SDK is ~50MB. For a Vercel serverless function
  cold start budget, raw httpx is dramatically faster.
"""

import asyncio
import json
import re
import time
import uuid
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.errors import ErrorCode, WorkflowError
from app.core.logging import get_logger, log_request_event
from app.models.schemas import WorkflowResult
from app.services.prompt import build_prompt
from app.models.schemas import ProcessRequest

logger = get_logger(__name__)
settings = get_settings()

# ── Gemini REST endpoint ───────────────────────────────────────────────────────
_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent?key={api_key}"
)


def _build_gemini_payload(system_prompt: str, user_message: str) -> dict:
    return {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_message}],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,        # low temp = more deterministic JSON output
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",  # Gemini 1.5 JSON mode
        },
    }


def _extract_json_text(response_body: dict) -> str:
    """
    Pull the raw text out of Gemini's response envelope.
    Raises WorkflowError on any structural anomaly.
    """
    try:
        candidates = response_body.get("candidates", [])
        if not candidates:
            raise WorkflowError(
                ErrorCode.LLM_EMPTY_RESPONSE,
                "Gemini returned no candidates.",
                internal=str(response_body),
            )

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            raise WorkflowError(
                ErrorCode.LLM_EMPTY_RESPONSE,
                "Gemini returned empty content parts.",
                internal=str(response_body),
            )

        text = parts[0].get("text", "").strip()
        if not text:
            raise WorkflowError(
                ErrorCode.LLM_EMPTY_RESPONSE,
                "Gemini returned blank text.",
                internal=str(response_body),
            )
        return text

    except WorkflowError:
        raise
    except Exception as exc:
        raise WorkflowError(
            ErrorCode.LLM_API_ERROR,
            "Unexpected Gemini response structure.",
            internal=str(exc),
        ) from exc


def _extract_token_usage(response_body: dict) -> dict[str, int]:
    """Pull token counts for logging. Non-fatal if missing."""
    meta = response_body.get("usageMetadata", {})
    return {
        "prompt_tokens": meta.get("promptTokenCount", 0),
        "output_tokens": meta.get("candidatesTokenCount", 0),
        "total_tokens": meta.get("totalTokenCount", 0),
    }


def _parse_json(raw: str) -> dict:
    """
    Strict JSON parse with fallback fence-stripping.
    LLMs sometimes wrap JSON in ```json ... ``` despite instructions.
    We strip that before failing hard.
    """
    # Fast path
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to extract first {...} block
    brace_match = re.search(r"\{[\s\S]+\}", raw)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise WorkflowError(
        ErrorCode.LLM_NON_JSON_RESPONSE,
        "The AI returned a non-JSON response. Please retry.",
        internal=f"raw_output={raw[:500]}",
    )


def _validate_schema(data: dict) -> WorkflowResult:
    """Run Pydantic validation. Maps validation errors to WorkflowError."""
    try:
        return WorkflowResult.model_validate(data)
    except Exception as exc:
        raise WorkflowError(
            ErrorCode.LLM_SCHEMA_INVALID,
            "The AI returned a response that does not match the required schema.",
            internal=str(exc),
        ) from exc


# ── Main async caller ─────────────────────────────────────────────────────────

async def call_gemini(
    request: ProcessRequest,
    request_id: str,
) -> tuple[WorkflowResult, dict[str, Any]]:
    """
    Calls Gemini with retry logic. Returns (WorkflowResult, meta_dict).
    meta_dict contains observability fields (token usage, latency, retries).

    Retry policy:
      - Max 2 retries (3 total attempts)
      - Retry on: timeout, non-JSON, schema invalid, API 5xx
      - No retry on: 4xx client errors (bad API key etc.)
    """
    if not settings.gemini_api_key:
        raise WorkflowError(
            ErrorCode.LLM_API_ERROR,
            "Service configuration error.",
            internal="GEMINI_API_KEY is not set.",
        )

    system_prompt, user_message = build_prompt(request)
    payload = _build_gemini_payload(system_prompt, user_message)
    url = _GEMINI_URL.format(
        model=settings.gemini_model,
        api_key=settings.gemini_api_key,
    )

    last_error: WorkflowError | None = None
    attempt = 0
    t_start = time.monotonic()

    async with httpx.AsyncClient(timeout=settings.gemini_timeout_seconds) as client:
        while attempt <= settings.max_retries:
            attempt += 1
            attempt_start = time.monotonic()

            try:
                log_request_event(
                    logger, request_id, "gemini_attempt",
                    attempt=attempt, max_attempts=settings.max_retries + 1,
                )

                response = await client.post(url, json=payload)

                if response.status_code == 429:
                    # Rate limited — worth retrying with backoff
                    raise WorkflowError(
                        ErrorCode.LLM_API_ERROR,
                        "AI service rate limit hit. Please retry in a moment.",
                        internal=f"status=429 body={response.text[:200]}",
                    )

                if response.status_code >= 500:
                    raise WorkflowError(
                        ErrorCode.LLM_API_ERROR,
                        "AI service unavailable.",
                        internal=f"status={response.status_code} body={response.text[:200]}",
                    )

                if response.status_code == 400:
                    raise WorkflowError(
                        ErrorCode.LLM_API_ERROR,
                        "Request rejected by AI service.",
                        internal=f"status=400 body={response.text[:300]}",
                    )

                if response.status_code not in (200, 201):
                    raise WorkflowError(
                        ErrorCode.LLM_API_ERROR,
                        "Unexpected response from AI service.",
                        internal=f"status={response.status_code}",
                    )

                body = response.json()
                token_usage = _extract_token_usage(body)
                raw_text = _extract_json_text(body)
                parsed = _parse_json(raw_text)
                result = _validate_schema(parsed)

                latency_ms = int((time.monotonic() - t_start) * 1000)
                attempt_ms = int((time.monotonic() - attempt_start) * 1000)

                log_request_event(
                    logger, request_id, "gemini_success",
                    attempt=attempt,
                    latency_ms=latency_ms,
                    attempt_latency_ms=attempt_ms,
                    validation_status="ok",
                    **token_usage,
                )

                return result, {
                    "retry_count": attempt - 1,
                    "latency_ms": latency_ms,
                    **token_usage,
                }

            except WorkflowError as exc:
                last_error = exc
                latency = int((time.monotonic() - attempt_start) * 1000)

                log_request_event(
                    logger, request_id, "gemini_attempt_failed",
                    attempt=attempt,
                    error_code=exc.code.value,
                    internal_detail=exc.internal,
                    attempt_latency_ms=latency,
                )

                # Don't retry client-side errors
                if exc.code in (ErrorCode.LLM_API_ERROR,) and attempt == 1:
                    # Check if it's a 4xx (non-retryable) by inspecting internal
                    if "status=400" in (exc.internal or "") or "status=401" in (exc.internal or ""):
                        break

                if attempt <= settings.max_retries:
                    await asyncio.sleep(settings.retry_delay_seconds * attempt)

            except httpx.TimeoutException as exc:
                last_error = WorkflowError(
                    ErrorCode.LLM_TIMEOUT,
                    "The AI service took too long to respond. Please try again.",
                    internal=str(exc),
                )
                latency = int((time.monotonic() - attempt_start) * 1000)
                log_request_event(
                    logger, request_id, "gemini_timeout",
                    attempt=attempt,
                    attempt_latency_ms=latency,
                )
                if attempt <= settings.max_retries:
                    await asyncio.sleep(settings.retry_delay_seconds * attempt)

            except Exception as exc:
                last_error = WorkflowError(
                    ErrorCode.INTERNAL_ERROR,
                    "An unexpected error occurred while calling the AI service.",
                    internal=str(exc),
                )
                break  # Unknown errors: fail fast, no retry

    # All retries exhausted
    total_ms = int((time.monotonic() - t_start) * 1000)
    log_request_event(
        logger, request_id, "gemini_exhausted",
        attempts=attempt,
        total_latency_ms=total_ms,
        final_error=last_error.code.value if last_error else "unknown",
    )

    raise WorkflowError(
        ErrorCode.RETRIES_EXHAUSTED,
        last_error.detail if last_error else "All retry attempts failed.",
        internal=last_error.internal if last_error else "",
    )
