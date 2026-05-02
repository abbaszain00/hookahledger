"""
main.py - FastAPI backend for HookahLedger.

Stage 1: sync /api/query endpoint. Streaming comes in stage 2.

Architecture:
  - Single AnswerEngine instance loaded at startup. BGE-M3 weights load
    once (~5s), not per-request. The engine is thread-safe for concurrent
    reads against ChromaDB and SQLite (per the audit).
  - CORS enabled for the Vite dev server (port 5173) during development.
    For production single-server deployment this could tighten to same-origin.

Run:
  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
from fastapi.responses import StreamingResponse

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from answer import AnswerEngine  # noqa: E402


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
    version="0.1.0",
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
# Request / response models
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
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    """Liveness check. Confirms the engine is loaded."""
    return {"status": "ok", "engine_loaded": _engine is not None}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Sync query endpoint. Returns the full AnswerResult as JSON.

    No streaming - that's stage 2. This endpoint is for the frontend to
    use during initial development before streaming is wired up, and
    also as a fallback path.
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

    # Flatten the dataclasses for JSON. asdict() handles nested dataclasses.
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

# ---------------------------------------------------------------------------
# SSE streaming endpoint
# ---------------------------------------------------------------------------
@app.get("/api/chat/stream")
def chat_stream(
    query: str,
    area: str | None = None,
    price_tier: str | None = None,
    aspect_positive: str | None = None,
    top_k: int = 5,
):
    """Stream Sonnet's response token-by-token via SSE, then send a final
    'evidence' event with the structured retrieval data (counts, lounges,
    chunks, validations) so the frontend can render evidence cards.

    SSE event shape:
      event: token
      data: <chunk text>

      event: evidence
      data: <json blob with lounges, chunks, validations, usage, cost>

      event: done
      data: {}

    The token events arrive as Sonnet generates them. The evidence event
    arrives after the stream completes and quote validation has run.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded yet")

    def event_stream():
        # We need to (a) stream tokens to the client AND (b) capture them so
        # we can run quote validation at the end. AnswerEngine._stream prints
        # to stdout; we replicate its logic here so we control the flow.

        # Run retrieval first - same as AnswerEngine.answer() does internally
        retrieval = _engine.retrieval.retrieve(
            query=query,
            area=area,
            price_tier=price_tier,
            aspect_positive=aspect_positive,
            top_k=top_k,
        )
        unique_lounge_ids = list({lg.lounge_id for lg in retrieval.lounges})
        verified_quotes = _engine.retrieval.fetch_verified_quotes(unique_lounge_ids)

        # Build the user message exactly as AnswerEngine does
        user_message = _engine._build_user_message(query, retrieval, verified_quotes)

        # Stream from Anthropic, forwarding each text delta as an SSE event
        chunks: list[str] = []
        tokens_in = 0
        tokens_out = 0
        try:
            with _engine.client.messages.stream(
                model=_engine.model,
                max_tokens=1500,
                system=__import__("answer").SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    chunks.append(text)
                    # SSE format: event name on its own line, then data, then blank line
                    yield f"event: token\ndata: {json.dumps(text)}\n\n"
                final = stream.get_final_message()
                if final.usage:
                    tokens_in = final.usage.input_tokens
                    tokens_out = final.usage.output_tokens
        except Exception as e:
            # If Sonnet errors mid-stream, send an error event and stop
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            return

        # Quote validation + evidence assembly
        full_text = "".join(chunks)
        from answer import neutralise_invalid_quotes  # local import to avoid circular issues
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
            "X-Accel-Buffering": "no",  # disable nginx buffering if behind proxy
        },
    )