
"""Single-process local UI/API: python -m uvicorn server:app --port 8000."""

import logging
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse

from rag_app.config import ROOT, Settings
from rag_app.schemas import Question, Response
from rag_app.service import Service
from rag_app.store import Busy, Conflict, Store


LOG_FILE = Path(os.getenv("LOG_FILE", "/tmp/server.log"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)

logger = logging.getLogger("rag_strategies")


def create_app(
    settings: Settings | None = None,
    store: Store | None = None,
    service: Service | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or Store(settings.database_path)
    service = service or Service(settings, store)

    logger.info(
        "Application initialized. database_path=%s log_file=%s",
        settings.database_path,
        LOG_FILE,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("Application startup complete.")

        try:
            yield
        finally:
            if service.engine is not None:
                logger.info("Closing retriever.")
                service.engine.retriever.close()

            logger.info("Application shutdown complete.")

    app = FastAPI(
        title="ChaiTech RAG Strategy Lab",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.service = service
    app.state.store = store

    @app.middleware("http")
    async def security_headers(
        request: Request,
        call_next,
    ):
        correlation_id = str(uuid.uuid4())

        logger.info(
            "Request started method=%s path=%s correlation_id=%s",
            request.method,
            request.url.path,
            correlation_id,
        )

        response = await call_next(request)

        response.headers.update(
            {
                "X-Correlation-ID": correlation_id,
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": (
                    "default-src 'self'; "
                    "script-src 'self'; "
                    "style-src 'self'; "
                    "frame-ancestors 'none'; "
                    "base-uri 'self'"
                ),
            }
        )

        if request.url.path in {
            "/docs",
            "/redoc",
            "/docs/oauth2-redirect",
        }:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: https://fastapi.tiangolo.com; "
                "frame-ancestors 'none'; "
                "base-uri 'self'"
            )

        logger.info(
            "Request complete method=%s path=%s status=%s correlation_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            correlation_id,
        )

        return response

    @app.exception_handler(Busy)
    @app.exception_handler(Conflict)
    async def conflict_handler(_: Request, error: Exception):
        logger.warning(
            "Storage conflict error_type=%s message=%s",
            type(error).__name__,
            str(error),
        )

        return JSONResponse(
            {"detail": str(error)},
            status_code=status.HTTP_409_CONFLICT,
        )

    @app.exception_handler(sqlite3.Error)
    async def database_error_handler(
        _: Request,
        error: sqlite3.Error,
    ):
        logger.exception(
            "Database error error_type=%s",
            type(error).__name__,
        )

        return JSONResponse(
            {
                "detail": (
                    "Local storage is unavailable. "
                    "Check the database path and server log."
                )
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/static/{name}", include_in_schema=False)
    def asset(name: str):
        allowed_assets = {"app.js", "styles.css"}

        if name not in allowed_assets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Asset not found.",
            )

        return FileResponse(ROOT / "static" / name)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "scope": "web process only; cloud services not probed",
        }

    @app.get("/api/status")
    def api_status():
        from langsmith.utils import tracing_is_enabled

        try:
            manifest = store.manifest()

            corpus = {
                key: manifest[key]
                for key in (
                    "index_name",
                    "namespace",
                    "sources",
                    "chunk_count",
                )
            }

        except ValueError:
            corpus = None

        return {
            "corpus": corpus,
            "pinecone_key_configured": bool(
                os.getenv("PINECONE_API_KEY")
            ),
            "tracing_enabled": bool(tracing_is_enabled()),
            "tracing_project": os.getenv("LANGSMITH_PROJECT"),
            "trace_delivery": (
                "Configuration only. "
                "Confirm remote delivery in LangSmith."
            ),
        }

    @app.post("/api/chat", response_model=Response)
    def chat(payload: Question):
        request_data = payload.model_dump()

        logger.info(
            "Chat request received request_id=%s session_id=%s source=%s",
            request_data.get("request_id"),
            request_data.get("session_id"),
            request_data.get("source"),
        )

        try:
            result = service.ask(request_data)

            logger.info(
                "Chat request completed request_id=%s",
                request_data.get("request_id"),
            )

            return result

        except (Busy, Conflict, sqlite3.Error):
            raise

        except ValueError as error:
            logger.exception(
                "Configuration/schema error request_id=%s error_type=%s",
                request_data.get("request_id"),
                type(error).__name__,
            )

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Check model/Pinecone configuration, run ingestion, "
                    "or retry the request."
                ),
            ) from error

        except Exception as error:
            logger.exception(
                "RAG dependency error request_id=%s error_type=%s",
                request_data.get("request_id"),
                type(error).__name__,
            )

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "A model or retrieval request failed. "
                    "Retry with the same request_id or check "
                    "service configuration."
                ),
            ) from error

    @app.delete("/api/sessions/{session_id}")
    def clear(session_id: str):
        store.clear(session_id)

        return {"cleared": session_id}

    return app


app = create_app()
