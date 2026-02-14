"""
AI Workflow Builder — Main Application Entry Point
FastAPI + Gemini + Pydantic | Serverless-friendly MVP
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AI Workflow Builder",
    description="Converts messy natural language instructions into structured execution plans.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Reshape Pydantic's validation errors to match our error contract."""
    errors = exc.errors()
    detail = "; ".join(
        f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
        for e in errors
    )
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", extra={"error": str(exc), "path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": "An unexpected error occurred."},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai-workflow-builder"}