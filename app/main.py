
"""
main.py - FastAPI backend for HookahLedger.

Two endpoint pairs:

  Bare RAG path (no agent):
    POST /api/query          - sync, returns full AnswerResult JSON
    GET  /api/chat/stream    - SSE: tokens + evidence event

  Agent path (LangGraph query understanding):
    POST /api/agent/query    - sync, runs the full graph
    GET  /api/agent/stream   - SSE: status + parsed + tokens + evidence

Architecture:
  - Single AnswerEngine instance loaded at startup. BGE-M3 weights load
    once (~5s), not per-request. The engine is thread-safe for concurrent
    reads against ChromaDB and SQLite (per the audit).
  - Agent nodes share the same retrieval pipeline / answer engine via the
    lazy singletons in agent/nodes.py.
  - CORS enabled for the Vite dev server (port 5173) during development.

Run:
  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Make scripts/ importable - must come before any `from answer import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from agent.graph import get_graph  # noqa: E402
from agent.nodes import (  # noqa: E402
    parse_query,
    validate_parse,
    retrieve_with_filters,
    retrieve_no_filter,
)
from agent.state import AgentState  # noqa: E402
from answer import (  # noqa: E402
    AnswerEngine,
    SYSTEM_PROMPT,
    neutralise_invalid_quotes,
)


# ---------------------------------------------------------------------------
# Lifespan: load the AnswerEngine once at startup
# ---------------------------------------------------------------------------
_engine: AnswerEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load AnswerEngine on startup, hold reference for the app lifetime."""
    global _engine
    print("Loading AnswerEngine (this includes BGE-M3, ~5s)...", file=sys.stderr)
    _engine = AnswerEngine()
    print("AnswerEngine ready.", file=sys.stderr)
    yield
    # No teardown needed - process exit cleans up


app = FastAPI(
    title="HookahLedger API",
    description="London shisha lounge intelligence engine",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS for Vite dev server. Keep it permissive in dev; tighten for prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models - bare path
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    area: Optional[str] = None
    price_tier: Optional[str] = None
    aspect_positive: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    """Mirror of AnswerResult, flattened for JSON."""
    query: str
    answer_raw: str
    answer_validated: str
    quote_validations: list[dict]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    lounges: list[dict]   # flattened LoungeEvidence (with chunks)
    chunks: list[dict]    # top-k retrieved chunks
    candidates_pulled: int


# ---------------------------------------------------------------------------
# Request / response models - agent path
# ---------------------------------------------------------------------------
class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class AgentQueryResponse(BaseModel):
    """Full final agent state plus the answer fields."""
    raw_query: str
    cleaned_query: Optional[str]
    area: Optional[str]
    price_tier: Optional[str]
    aspect_positive: Optional[str]
    parse_confidence: float
    parse_valid: Optional[bool]
    validation_reason: Optional[str]
    used_filters: bool
    answer_raw: str
    answer_validated: str
    quote_validations: list[dict]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    lounges: list[dict]
    chunks: list[dict]
    candidates_pulled: int


# ---------------------------------------------------------------------------
# Bare endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    """Liveness check. Confirms the engine is loaded."""
    return {"status": "ok", "engine_loaded": _engine is not None}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Sync query endpoint. Returns the full AnswerResult as JSON.

    No streaming, no agent. The frontend uses this as a fallback path
    and the eval harness uses it for batch scoring.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded yet")

    result = _engine.answer(
        query=req.query,
        area=req.area,
        price_tier=req.price_tier,
        aspect_positive=req.aspect_positive,
        top_k=req.top_k,
        stream=False,
    )

    return QueryResponse(
        query=result.query,
        answer_raw=result.text,
        answer_validated=result.text_validated,
        quote_validations=[asdict(v) for v in result.quote_validations],
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_usd=result.cost_usd,
        lounges=[
            {
                **{k: v for k, v in asdict(lg).items() if k != "chunks"},
                "chunks": [asdict(c) for c in lg.chunks],
            }
            for lg in result.retrieval.lounges
        ],
        chunks=[asdict(c) for c in result.retrieval.chunks],
        candidates_pulled=result.retrieval.candidates_pulled,
    )


@app.get("/api/chat/stream")
def chat_stream(
    query: str,
    area: str | None = None,
    price_tier: str | None = None,
    aspect_positive: str | None = None,
    top_k: int = 5,
):
    """Stream Sonnet's response token-by-token via SSE, then send a final
    'evidence' event with the structured retrieval data.

    SSE event shape:
      event: token       data: <chunk text>
      event: error       data: {"error": "..."}        (only on stream failure)
      event: evidence    data: <json blob, includes degraded:bool>
      event: done        data: {}

    Validation runs in a finally block so a stream error never bypasses
    neutralise_invalid_quotes. The evidence event always fires (with
    degraded:true on failure) so the frontend can render whatever partial
    answer was received with hallucinated quotes neutralised.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded yet")

    def event_stream():
        retrieval = _engine.retrieval.retrieve(
            query=query,
            area=area,
            price_tier=price_tier,
            aspect_positive=aspect_positive,
            top_k=top_k,
        )
        unique_lounge_ids = list({lg.lounge_id for lg in retrieval.lounges})
        verified_quotes = _engine.retrieval.fetch_verified_quotes(unique_lounge_ids)

        user_message = _engine._build_user_message(query, retrieval, verified_quotes)

        chunks: list[str] = []
        tokens_in = 0
        tokens_out = 0
        stream_failed = False
        try:
            with _engine.client.messages.stream(
                model=_engine.model,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    chunks.append(text)
                    yield f"event: token\ndata: {json.dumps(text)}\n\n"
                final = stream.get_final_message()
                if final.usage:
                    tokens_in = final.usage.input_tokens
                    tokens_out = final.usage.output_tokens
        except Exception as e:
            stream_failed = True
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        finally:
            full_text = "".join(chunks)
            text_validated, validations = neutralise_invalid_quotes(full_text, verified_quotes)
            cost_usd = (tokens_in / 1_000_000) * 3.0 + (tokens_out / 1_000_000) * 15.0

            evidence = {
                "answer_validated": text_validated,
                "quote_validations": [
                    {"quote": v.quote, "valid": v.valid} for v in validations
                ],
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost_usd,
                "degraded": stream_failed,
                "rerank_succeeded": retrieval.rerank_succeeded,
                "lounges": [
                    {
                        **{k: v for k, v in asdict(lg).items() if k != "chunks"},
                        "chunks": [asdict(c) for c in lg.chunks],
                    }
                    for lg in retrieval.lounges
                ],
                "chunks": [asdict(c) for c in retrieval.chunks],
                "candidates_pulled": retrieval.candidates_pulled,
            }
            yield f"event: evidence\ndata: {json.dumps(evidence)}\n\n"
            yield f"event: done\ndata: {{}}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Agent endpoints (LangGraph query understanding)
# ---------------------------------------------------------------------------
@app.post("/api/agent/query", response_model=AgentQueryResponse)
def agent_query(req: AgentQueryRequest) -> AgentQueryResponse:
    """Sync agent endpoint. Runs the full graph (parse -> validate -> retrieve
    -> generate) via graph.invoke() and returns the final state as JSON.

    The graph is built once at module load via get_graph() and cached.
    Caches inside the nodes ensure the retrieval pipeline and answer engine
    are reused across requests.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded yet")

    graph = get_graph()
    initial = AgentState(raw_query=req.query)
    final_state = graph.invoke(initial)

    # graph.invoke() returns a dict matching AgentState's field shape
    answer = final_state.get("answer_result")
    retrieval = final_state.get("retrieval_result")
    if answer is None or retrieval is None:
        raise HTTPException(
            status_code=500,
            detail="Agent graph completed without producing an answer",
        )

    return AgentQueryResponse(
        raw_query=final_state["raw_query"],
        cleaned_query=final_state.get("cleaned_query"),
        area=final_state.get("area"),
        price_tier=final_state.get("price_tier"),
        aspect_positive=final_state.get("aspect_positive"),
        parse_confidence=final_state.get("parse_confidence", 0.0),
        parse_valid=final_state.get("parse_valid"),
        validation_reason=final_state.get("validation_reason"),
        used_filters=final_state.get("used_filters", False),
        answer_raw=answer.text,
        answer_validated=answer.text_validated,
        quote_validations=[asdict(v) for v in answer.quote_validations],
        tokens_in=answer.tokens_in,
        tokens_out=answer.tokens_out,
        cost_usd=answer.cost_usd,
        lounges=[
            {
                **{k: v for k, v in asdict(lg).items() if k != "chunks"},
                "chunks": [asdict(c) for c in lg.chunks],
            }
            for lg in retrieval.lounges
        ],
        chunks=[asdict(c) for c in retrieval.chunks],
        candidates_pulled=retrieval.candidates_pulled,
    )


@app.get("/api/agent/stream")
def agent_stream(query: str, top_k: int = 5):
    """Streaming agent endpoint with status events.

    Event sequence:
      event: status     data: {"phase": "parsing", "message": "..."}
      event: parsed     data: {"area", "price_tier", "aspect_positive",
                               "cleaned_query", "confidence", "parse_valid",
                               "validation_reason"}
      event: status     data: {"phase": "retrieving", "message": "..."}
      event: status     data: {"phase": "generating", "message": "..."}
      event: token      data: <chunk>      (many)
      event: evidence   data: {... structured retrieval + validation + agent ...}
      event: done       data: {}

    Implemented imperatively (not via LangGraph .invoke) because we need to
    interleave SSE events between each node's execution. Same node functions
    as the graph; only the orchestration differs.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded yet")

    def event_stream():
        # Phase 1: parse
        yield (
            "event: status\ndata: "
            + json.dumps({"phase": "parsing", "message": "Understanding your query..."})
            + "\n\n"
        )
        state = AgentState(raw_query=query)
        parse_update = parse_query(state)
        for k, v in parse_update.items():
            setattr(state, k, v)

        # Phase 2: validate
        validate_update = validate_parse(state)
        for k, v in validate_update.items():
            setattr(state, k, v)

        yield (
            "event: parsed\ndata: "
            + json.dumps({
                "area": state.area,
                "price_tier": state.price_tier,
                "aspect_positive": state.aspect_positive,
                "cleaned_query": state.cleaned_query,
                "confidence": state.parse_confidence,
                "parse_valid": state.parse_valid,
                "validation_reason": state.validation_reason,
            })
            + "\n\n"
        )

        # Phase 3: retrieval (filtered or fallback)
        if state.parse_valid:
            filters_summary = ", ".join(
                f"{k}={v}" for k, v in {
                    "area": state.area,
                    "price_tier": state.price_tier,
                    "aspect_positive": state.aspect_positive,
                }.items() if v
            ) or "no filters"
            yield (
                "event: status\ndata: "
                + json.dumps({
                    "phase": "retrieving",
                    "message": f"Searching with filters: {filters_summary}",
                })
                + "\n\n"
            )
            retrieval_update = retrieve_with_filters(state)
        else:
            yield (
                "event: status\ndata: "
                + json.dumps({
                    "phase": "retrieving",
                    "message": f"Falling back to unfiltered search ({state.validation_reason})",
                })
                + "\n\n"
            )
            retrieval_update = retrieve_no_filter(state)
        for k, v in retrieval_update.items():
            setattr(state, k, v)

        retrieval = state.retrieval_result
        if retrieval is None:
            yield (
                "event: error\ndata: "
                + json.dumps({"error": "Retrieval failed"})
                + "\n\n"
            )
            return

        # Phase 4: stream Sonnet tokens
        yield (
            "event: status\ndata: "
            + json.dumps({"phase": "generating", "message": "Generating answer..."})
            + "\n\n"
        )

        unique_lounge_ids = list({lg.lounge_id for lg in retrieval.lounges})
        verified_quotes = _engine.retrieval.fetch_verified_quotes(unique_lounge_ids)

        # Use raw_query as the "Question:" line so the model sees what the
        # user actually asked, not the agent-stripped version.
        user_message = _engine._build_user_message(
            state.raw_query, retrieval, verified_quotes
        )

        chunks: list[str] = []
        tokens_in = 0
        tokens_out = 0
        stream_failed = False
        try:
            with _engine.client.messages.stream(
                model=_engine.model,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    chunks.append(text)
                    yield f"event: token\ndata: {json.dumps(text)}\n\n"
                final = stream.get_final_message()
                if final.usage:
                    tokens_in = final.usage.input_tokens
                    tokens_out = final.usage.output_tokens
        except Exception as e:
            stream_failed = True
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        finally:
            # Phase 5: quote validation + evidence event (runs on success and failure)
            full_text = "".join(chunks)
            text_validated, validations = neutralise_invalid_quotes(full_text, verified_quotes)
            cost_usd = (tokens_in / 1_000_000) * 3.0 + (tokens_out / 1_000_000) * 15.0

            evidence = {
                "answer_validated": text_validated,
                "quote_validations": [
                    {"quote": v.quote, "valid": v.valid} for v in validations
                ],
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost_usd,
                "degraded": stream_failed,
                "rerank_succeeded": retrieval.rerank_succeeded,
                "lounges": [
                    {
                        **{k: v for k, v in asdict(lg).items() if k != "chunks"},
                        "chunks": [asdict(c) for c in lg.chunks],
                    }
                    for lg in retrieval.lounges
                ],
                "chunks": [asdict(c) for c in retrieval.chunks],
                "candidates_pulled": retrieval.candidates_pulled,
                "agent": {
                    "area": state.area,
                    "price_tier": state.price_tier,
                    "aspect_positive": state.aspect_positive,
                    "cleaned_query": state.cleaned_query,
                    "parse_confidence": state.parse_confidence,
                    "parse_valid": state.parse_valid,
                    "validation_reason": state.validation_reason,
                    "used_filters": state.used_filters,
                },
            }
            yield f"event: evidence\ndata: {json.dumps(evidence)}\n\n"
            yield f"event: done\ndata: {{}}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )