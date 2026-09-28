from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from bumi_mem.long_term import resolve_embed_device, warmup_mem0_backend

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("mem0-locomo-server")

ROOT = Path(
    os.getenv(
        "MEM0_REPRO_ROOT",
        str(Path(__file__).resolve().parent.parent),
    )
).resolve()
RUNTIME_ROOT = Path(os.getenv("MEM0_RUNTIME_ROOT", str(ROOT.parent / "mem0-runtime"))).resolve()
MODEL_ROOT = Path(os.getenv("MEM0_MODEL_ROOT", str(ROOT.parent / "models"))).resolve()
DATA_DIR = Path(os.getenv("MEM0_DATA_DIR", str(RUNTIME_ROOT / "data" / "locomo_v3" / "default"))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

_memory: Any | None = None
_memory_lock = threading.Lock()
_warmup_elapsed_ms: float | None = None


class AddRequest(BaseModel):
    messages: list[dict[str, str]]
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    timestamp: int | float | str | None = None
    custom_instructions: str | None = None
    metadata: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    filters: dict[str, Any] | None = None
    limit: int = Field(default=200, ge=1, le=1000)
    top_k: int | None = Field(default=None, ge=1, le=1000)
    threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    rerank: bool = False
    explain: bool = False
    reference_date: int | float | str | None = None


def _settings() -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or DEEPSEEK_API_KEY is required")

    base_url = (
        os.getenv("MEMORY_API_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com/v1"
    ).rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    embed_model = Path(
        os.getenv("MEM0_EMBED_MODEL", str(MODEL_ROOT / "Qwen3-Embedding-0.6B"))
    ).resolve()
    if not embed_model.exists():
        raise RuntimeError(f"Embedding model does not exist: {embed_model}")

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "embed_model": embed_model,
        "embed_device": resolve_embed_device(os.getenv("MEM0_EMBED_DEVICE", "auto")),
        "collection": os.getenv("MEM0_COLLECTION_NAME", "locomo_v3_memories"),
        "memory_max_tokens": int(os.getenv("MEM0_MEMORY_MAX_TOKENS", "4096")),
    }


def _get_memory():
    global _memory
    if _memory is not None:
        return _memory

    with _memory_lock:
        if _memory is not None:
            return _memory

        from mem0 import Memory

        s = _settings()
        config = {
            # This selects Mem0's v1.1 response envelope. The installed package
            # is Mem0 2.x and uses the 2026 ADD-only extraction pipeline.
            "version": "v1.1",
            "history_db_path": str(DATA_DIR / "history.db"),
            "llm": {
                "provider": "openai",
                "config": {
                    "model": s["model"],
                    "api_key": s["api_key"],
                    "openai_base_url": s["base_url"],
                    "temperature": 0.0,
                    "max_tokens": s["memory_max_tokens"],
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": str(s["embed_model"]),
                    "embedding_dims": 1024,
                    "model_kwargs": {
                        "device": s["embed_device"],
                        "local_files_only": True,
                    },
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": s["collection"],
                    "embedding_model_dims": 1024,
                    "path": str(DATA_DIR / "qdrant"),
                    "on_disk": True,
                },
            },
        }
        logger.info(
            "Initializing Mem0: package=2.0.20 model=%s embedder=%s data=%s",
            s["model"],
            s["embed_model"],
            DATA_DIR,
        )
        _memory = Memory.from_config(config)
        logger.info("Mem0 initialization complete")
        return _memory


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _startup_warmup() -> float | None:
    global _warmup_elapsed_ms
    if not _env_flag("BUMI_MEMORY_WARMUP", True):
        logger.info("Mem0 startup warmup disabled")
        return None
    if _warmup_elapsed_ms is not None:
        return _warmup_elapsed_ms
    query = os.getenv("BUMI_MEMORY_WARMUP_QUERY", "记忆系统预热")
    started = time.perf_counter()
    memory = _get_memory()
    warmup_mem0_backend(memory, query=query)
    _warmup_elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Mem0 startup warmup complete: %.3f ms device=%s",
        _warmup_elapsed_ms,
        _settings()["embed_device"],
    )
    return _warmup_elapsed_ms


@asynccontextmanager
async def lifespan(_: FastAPI):
    _startup_warmup()
    yield


app = FastAPI(
    title="Mem0 v2.0.20 LoCoMo OSS server",
    version="2.0.20",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    from importlib.metadata import version

    return {
        "status": "ok",
        "mem0ai": version("mem0ai"),
        "algorithm": "2026-add-only",
        "initialized": _memory is not None,
        "warmed_up": _warmup_elapsed_ms is not None,
        "warmup_elapsed_ms": _warmup_elapsed_ms,
        "embed_device": resolve_embed_device(os.getenv("MEM0_EMBED_DEVICE", "auto")),
        "data_dir": str(DATA_DIR),
    }


@app.post("/memories")
def add_memories(req: AddRequest):
    mem = _get_memory()
    params: dict[str, Any] = {}
    for key in ("user_id", "agent_id", "run_id"):
        value = getattr(req, key)
        if value is not None:
            params[key] = value

    # Released Mem0 OSS 2.0.20 is ADD-only but its native timestamp argument is
    # Platform-only. Preserve LoCoMo session time as metadata and explicit
    # extraction context instead of passing the unsupported timestamp argument.
    metadata = dict(req.metadata or {})
    temporal_instruction = None
    if req.timestamp is not None:
        try:
            dt = datetime.fromtimestamp(float(req.timestamp), tz=timezone.utc)
            observation_date = dt.strftime("%Y-%m-%d")
            metadata.setdefault("observation_date", observation_date)
            metadata.setdefault("source_epoch", req.timestamp)
            temporal_instruction = (
                f"This conversation occurred on {observation_date}. "
                "Resolve relative dates against that observation date and "
                "preserve explicit dates in extracted memories."
            )
        except (TypeError, ValueError, OSError):
            logger.warning("Ignoring invalid timestamp: %r", req.timestamp)
    if metadata:
        params["metadata"] = metadata

    instructions = [x for x in (req.custom_instructions, temporal_instruction) if x]
    if instructions:
        params["prompt"] = " ".join(instructions)
    try:
        return mem.add(req.messages, **params)
    except Exception as exc:
        logger.exception("Mem0 add failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search")
def search_memories(req: SearchRequest):
    mem = _get_memory()
    filters = dict(req.filters or {})
    for key in ("user_id", "agent_id", "run_id"):
        value = getattr(req, key)
        if value is not None:
            filters[key] = value
    if not filters:
        raise HTTPException(status_code=400, detail="At least one entity filter is required")
    try:
        return mem.search(
            req.query,
            filters=filters,
            top_k=req.top_k or req.limit,
            threshold=req.threshold,
            rerank=req.rerank,
            explain=req.explain,
        )
    except Exception as exc:
        logger.exception("Mem0 search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/memories")
def get_memories(
    user_id: str | None = Query(None),
    agent_id: str | None = Query(None),
    run_id: str | None = Query(None),
    top_k: int = Query(1000, ge=1, le=10000),
):
    filters = {k: v for k, v in {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}.items() if v}
    if not filters:
        raise HTTPException(status_code=400, detail="Provide user_id, agent_id, or run_id")
    try:
        return _get_memory().get_all(filters=filters, top_k=top_k)
    except Exception as exc:
        logger.exception("Mem0 get_all failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/memories")
def delete_memories(
    user_id: str | None = Query(None),
    agent_id: str | None = Query(None),
    run_id: str | None = Query(None),
):
    if not any((user_id, agent_id, run_id)):
        raise HTTPException(status_code=400, detail="Provide user_id, agent_id, or run_id")
    try:
        return _get_memory().delete_all(user_id=user_id, agent_id=agent_id, run_id=run_id)
    except Exception as exc:
        logger.exception("Mem0 delete_all failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
