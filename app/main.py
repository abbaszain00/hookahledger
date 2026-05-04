"""
main.py - FastAPI backend for HookahLedger.

Two endpoint pairs:

  Bare RAG path (no agent):
    POST /api/query          - sync, returns full AnswerResult JSON
    GET  /api/chat/stream    - SSE: tokens + evidence event

  Agent path (LangGraph query understanding):
    POST /api/agent/query    - sync, runs the full graph
    GET  /api/agent/stream   - SSE: status + parsed + tokens + evidence

Session memory:
  Both stream endpoints accept an optional session_id query param. Sessions
  hold three things, each persisted on a successful turn:

    1. Conversation messages for Sonnet (user question text + validated
       answer text, no evidence block) so the model sees the conversational
       arc.
    2. Last parsed filters (area, price_tier, aspect_positive, lounge_focus)
       so Haiku can inherit / merge them on the next turn.
    3. Last retrieved lounge IDs in order, so Haiku can resolve references
       like "tell me more about the first one".

  Memory is opt-in: omit session_id and the system behaves as single-turn.
  Sessions are in-memory and reset on server restart - sufficient for a
  single-judge demo.

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
from threading import Lock
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
    decline_query,
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
    version="0.4.0",
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
# Session memory
# ---------------------------------------------------------------------------
# Each session holds:
#   messages: list of {role, content} for Sonnet (prior conversational turns)
#   last_filters: {area, price_tier, aspect_positive, lounge_focus} or None
#   last_results: list of lounge_ids in retrieval order, or empty list
#
# All three are populated on every successful turn. None / empty on a fresh
# session. The session as a whole is opt-in via the session_id query param;
# omit it and the endpoints behave exactly single-turn.
#
# Lock guards concurrent reads/writes. FastAPI runs sync handlers in a
# thread pool so we'd otherwise have a race on simultaneous requests for
# the same session_id.
_sessions: dict[str, dict] = {}
_sessions_lock = Lock()

# Cap on prior message turns retained for Sonnet. Beyond this, the oldest
# pair is dropped. Each turn = 2 messages (~500-800 tokens for the answer
# plus ~20 for the question). Cap of 10 keeps prior history bounded at
# roughly 5-8K tokens, well clear of context limits.
MAX_PRIOR_TURNS = 10


def _empty_session() -> dict:
    return {
        "messages": [],
        "last_filters": None,
        "last_results": [],
    }


def get_prior_messages(session_id: str | None) -> list[dict]:
    """Return prior {role, content} messages for a session, or empty list."""
    if not session_id:
        return []
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if not sess:
            return []
        return list(sess["messages"])


def get_prior_filters(session_id: str | None) -> dict | None:
    """Return last parsed filters for a session, or None.

    Shape: {area, price_tier, aspect_positive, lounge_focus} or None on a
    fresh session. Returned as a copy so callers can mutate freely.
    """
    if not session_id:
        return None
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if not sess or not sess["last_filters"]:
            return None
        return dict(sess["last_filters"])


def get_prior_results(session_id: str | None) -> list[str]:
    """Return last retrieval's lounge_ids in order, or empty list."""
    if not session_id:
        return []
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if not sess:
            return []
        return list(sess["last_results"])


def append_turn(
    session_id: str | None,
    user_question: str,
    assistant_answer: str,
    filters: dict | None = None,
    lounge_ids: list[str] | None = None,
) -> None:
    """Append a completed turn to session memory.

    No-op if session_id is None. Trims oldest message pairs if past
    MAX_PRIOR_TURNS. Replaces last_filters / last_results wholesale -
    they reflect the most recent turn's state, not an accumulation.
    """
    if not session_id:
        return
    with _sessions_lock:
        sess = _sessions.setdefault(session_id, _empty_session())
        sess["messages"].append({"role": "user", "content": user_question})
        sess["messages"].append({"role": "assistant", "content": assistant_answer})
        # Trim oldest pairs past the cap
        max_messages = MAX_PRIOR_TURNS * 2
        if len(sess["messages"]) > max_messages:
            sess["messages"] = sess["messages"][-max_messages:]
        # Replace filters and results - they're snapshots, not history
        if filters is not None:
            sess["last_filters"] = filters
        if lounge_ids is not None:
            sess["last_results"] = lounge_ids


@app.post("/api/session/reset")
def session_reset(session_id: str) -> dict:
    """Clear all state for a session. Frontend hits this on
    'New conversation' click and on mode switch."""
    with _sessions_lock:
        _sessions.pop(session_id, None)
    return {"ok": True, "session_id": session_id}


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
    lounges: list[dict]
    chunks: list[dict]
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
    """Sync query endpoint. No streaming, no agent, no session memory.
    The frontend uses this as a fallback path and the eval harness uses
    it for batch scoring.
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
    session_id: str | None = None,
):
    """Stream Sonnet's response token-by-token via SSE, then send a final
    'evidence' event with the structured retrieval data.

    Bare path - no parser merge logic. If session_id is provided, prior
    messages are prepended to Sonnet's messages list so the model sees the
    conversation arc, but filters are not inherited (the user supplied
    them explicitly via dropdowns).

    Validation runs in a finally block so a stream error never bypasses
    neutralise_invalid_quotes.
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

        prior = get_prior_messages(session_id)
        messages = prior + [{"role": "user", "content": user_message}]

        chunks: list[str] = []
        tokens_in = 0
        tokens_out = 0
        stream_failed = False
        try:
            with _engine.client.messages.stream(
                model=_engine.model,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=messages,
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

            # Persist the turn. Bare path stores the user-supplied filters
            # as the "last filters" so a future agent-mode turn could pick
            # them up if the user mode-switched (we currently reset on mode
            # switch, but storing this costs nothing).
            if not stream_failed and full_text.strip():
                bare_filters = {
                    "area": area,
                    "price_tier": price_tier,
                    "aspect_positive": aspect_positive,
                    "lounge_focus": None,
                }
                lounge_ids_in_order = [
                    lg.lounge_id for lg in retrieval.lounges
                ]
                append_turn(
                    session_id,
                    query,
                    text_validated,
                    filters=bare_filters,
                    lounge_ids=lounge_ids_in_order,
                )

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
    """Sync agent endpoint. Single-turn, no session memory. Used by the
    eval harness which needs deterministic single-turn behaviour."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded yet")

    graph = get_graph()
    initial = AgentState(raw_query=req.query)
    final_state = graph.invoke(initial)

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
def agent_stream(query: str, top_k: int = 5, session_id: str | None = None):
    """Streaming agent endpoint with status events.

    Event sequence:
      event: status     data: {"phase": "parsing", "message": "..."}
      event: parsed     data: {"area", "price_tier", "aspect_positive",
                               "lounge_focus", "cleaned_query",
                               "confidence", "parse_valid",
                               "validation_reason", "inherited_filters"}
      event: status     data: {"phase": "retrieving", "message": "..."}
      event: status     data: {"phase": "generating", "message": "..."}
      event: token      data: <chunk>      (many)
      event: evidence   data: {... structured retrieval + validation + agent ...}
      event: done       data: {}

    Session memory: if session_id is provided, prior messages, filters, and
    last results are loaded for the parser to consider. After a successful
    turn the session is updated with the new filters and lounge order.

    Off-taxonomy fast-decline path: when parse_query flags is_in_taxonomy=False,
    skip retrieval and Sonnet entirely. Decline turns do not update the
    session.

    Implemented imperatively (not via LangGraph .invoke) because we need to
    interleave SSE events between each node's execution. Same node functions
    as the graph; only the orchestration differs.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded yet")

    def event_stream():
        # Phase 1: parse with optional prior context
        yield (
            "event: status\ndata: "
            + json.dumps({"phase": "parsing", "message": "Understanding your query..."})
            + "\n\n"
        )

        # Build initial state with prior context. parse_query reads these to
        # decide whether to use the merge prompt or the standard prompt.
        prior_filters = get_prior_filters(session_id)
        prior_results = get_prior_results(session_id)

        state = AgentState(
            raw_query=query,
            prior_filters=prior_filters,
            prior_results=prior_results,
        )

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
                "lounge_focus": state.lounge_focus,
                "cleaned_query": state.cleaned_query,
                "confidence": state.parse_confidence,
                "parse_valid": state.parse_valid,
                "validation_reason": state.validation_reason,
                "inherited_filters": sorted(state.inherited_filters or []),
            })
            + "\n\n"
        )

        # Phase 2b: fast-decline if the parser flagged the query as off-taxonomy.
        if state.is_in_taxonomy is False:
            decline_update = decline_query(state)
            for k, v in decline_update.items():
                setattr(state, k, v)

            decline_text = state.answer_result.text
            yield f"event: token\ndata: {json.dumps(decline_text)}\n\n"

            evidence = {
                "answer_validated": decline_text,
                "quote_validations": [],
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0.0,
                "degraded": False,
                "rerank_succeeded": True,
                "is_declined": True,
                "decline_reason": state.decline_reason,
                "lounges": [],
                "chunks": [],
                "candidates_pulled": 0,
                "agent": {
                    "area": state.area,
                    "price_tier": state.price_tier,
                    "aspect_positive": state.aspect_positive,
                    "lounge_focus": state.lounge_focus,
                    "cleaned_query": state.cleaned_query,
                    "parse_confidence": state.parse_confidence,
                    "parse_valid": state.parse_valid,
                    "validation_reason": state.validation_reason,
                    "used_filters": False,
                    "inherited_filters": sorted(state.inherited_filters or []),
                },
            }
            yield f"event: evidence\ndata: {json.dumps(evidence)}\n\n"
            yield f"event: done\ndata: {{}}\n\n"
            return

        # Phase 3: retrieval (filtered or fallback)
        if state.parse_valid:
            filters_summary_parts = []
            if state.lounge_focus:
                filters_summary_parts.append(f"lounge={state.lounge_focus}")
            if state.area:
                filters_summary_parts.append(f"area={state.area}")
            if state.price_tier:
                filters_summary_parts.append(f"price={state.price_tier}")
            if state.aspect_positive:
                filters_summary_parts.append(f"aspect={state.aspect_positive}")
            filters_summary = ", ".join(filters_summary_parts) or "no filters"
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

        prior_messages = get_prior_messages(session_id)
        messages = prior_messages + [{"role": "user", "content": user_message}]

        chunks: list[str] = []
        tokens_in = 0
        tokens_out = 0
        stream_failed = False
        try:
            with _engine.client.messages.stream(
                model=_engine.model,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=messages,
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
            # Phase 5: quote validation + evidence event
            full_text = "".join(chunks)
            text_validated, validations = neutralise_invalid_quotes(full_text, verified_quotes)
            cost_usd = (tokens_in / 1_000_000) * 3.0 + (tokens_out / 1_000_000) * 15.0

            # Persist the turn to the session if the stream succeeded.
            # Snapshot the merged filter state and the lounge order for the
            # next turn's parser to consider.
            if not stream_failed and full_text.strip():
                turn_filters = {
                    "area": state.area,
                    "price_tier": state.price_tier,
                    "aspect_positive": state.aspect_positive,
                    "lounge_focus": state.lounge_focus,
                }
                lounge_ids_in_order = [lg.lounge_id for lg in retrieval.lounges]
                append_turn(
                    session_id,
                    state.raw_query,
                    text_validated,
                    filters=turn_filters,
                    lounge_ids=lounge_ids_in_order,
                )

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
                "is_declined": False,
                "decline_reason": None,
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
                    "lounge_focus": state.lounge_focus,
                    "cleaned_query": state.cleaned_query,
                    "parse_confidence": state.parse_confidence,
                    "parse_valid": state.parse_valid,
                    "validation_reason": state.validation_reason,
                    "used_filters": state.used_filters,
                    "inherited_filters": sorted(state.inherited_filters or []),
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