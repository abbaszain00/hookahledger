"""
nodes.py - LangGraph node functions for the query understanding agent.

Four nodes:
  parse_query       - Haiku tool-use call to extract structured filters
  validate_parse    - Pure Python schema check; sets parse_valid for routing
  retrieve_with_filters / retrieve_no_filter - Two retrieval paths
  generate_answer   - Reuses the existing AnswerEngine for the Sonnet call

Each node receives the current AgentState and returns a dict of fields to
update. LangGraph merges those into the state for the next node.

Design notes:
  - parse_query uses Anthropic's native tool use, not LangChain's structured
    output abstractions. The schema is enforced by the API: Haiku either
    returns a tool_use block matching ParsedQuery, or an error we catch.
  - The retrieval and generation nodes wrap the existing pipeline rather
    than reimplementing it. The agent layer is additive, not replacement.
  - Pipeline / engine instances are loaded lazily and cached at module level
    so we don't reload BGE-M3 per request.
"""

from __future__ import annotations

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import sys
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

# Reuse the retrieval pipeline and answer engine from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from retrieve import RetrievalPipeline  # noqa: E402
from answer import AnswerEngine  # noqa: E402

from agent.state import (  # noqa: E402
    AgentState,
    ParsedQuery,
    VALID_AREAS,
    VALID_PRICE_TIERS,
    VALID_ASPECTS,
)


# ---------------------------------------------------------------------------
# Lazy singletons - loaded on first use, kept across requests
# ---------------------------------------------------------------------------
_anthropic_client: anthropic.Anthropic | None = None
_retrieval_pipeline: RetrievalPipeline | None = None
_answer_engine: AnswerEngine | None = None


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        project_root = Path(__file__).resolve().parent.parent
        load_dotenv(project_root / ".env")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not in .env")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _get_retrieval_pipeline() -> RetrievalPipeline:
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        _retrieval_pipeline = RetrievalPipeline()
    return _retrieval_pipeline


def _get_answer_engine() -> AnswerEngine:
    global _answer_engine
    if _answer_engine is None:
        # Reuse the cached retrieval pipeline rather than letting AnswerEngine
        # build its own (which would load BGE-M3 a second time).
        _answer_engine = AnswerEngine(retrieval_pipeline=_get_retrieval_pipeline())
    return _answer_engine


# ---------------------------------------------------------------------------
# Node 1: parse_query
# ---------------------------------------------------------------------------
PARSE_TOOL_DEFINITION = {
    "name": "extract_query_filters",
    "description": (
        "Extract structured filters from a natural-language query about "
        "London shisha lounges. Identify the area, price tier, and main "
        "aspect being asked about. Return a cleaned semantic query for "
        "vector retrieval."
    ),
    "input_schema": ParsedQuery.model_json_schema(),
}

PARSE_SYSTEM_PROMPT = """You are a query understanding assistant for a London shisha lounge search system. Your job is to read a natural-language query and extract structured filters.

The query may contain:
- An area or neighbourhood (Soho, Edgware Road, Kingsbury, etc)
- A price constraint (under £20, premium, cheap, etc)
- A specific aspect being asked about (good service, smooth shisha, food quality, etc)
- Or none of these - a general "best lounge" query

Use the extract_query_filters tool to return your parse. Always call the tool exactly once.

Be conservative: if you're not sure whether the query implies a filter, return null for that field rather than guessing. Confidence below 0.5 means the query is too vague to filter on - the system will fall back to unfiltered retrieval."""


def parse_query(state: AgentState) -> dict[str, Any]:
    """Call Haiku with the extract_query_filters tool to parse the user's query.

    Returns a state update with the parsed fields, or sets parse_error if
    the call failed. Does NOT raise - failure is a routable state, not an
    exception.
    """
    client = _get_anthropic_client()

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            temperature=0,
            system=PARSE_SYSTEM_PROMPT,
            tools=[PARSE_TOOL_DEFINITION],
            tool_choice={"type": "tool", "name": "extract_query_filters"},
            messages=[{"role": "user", "content": state.raw_query}],
        )
    except Exception as e:
        # API error - log it, fall back to unfiltered. Validation node will
        # see parse_error set and route accordingly.
        return {
            "parse_error": f"{type(e).__name__}: {e}",
            "parse_confidence": 0.0,
        }

    # tool_choice='tool' guarantees the model used the tool, but defensively
    # walk the content blocks looking for it.
    tool_block = next(
        (b for b in response.content if b.type == "tool_use"
         and b.name == "extract_query_filters"),
        None,
    )
    if tool_block is None:
        return {
            "parse_error": "Model did not return the expected tool call",
            "parse_confidence": 0.0,
        }

    # Pydantic validates the structure; the API already enforced the schema
    # but we double-check for safety.
    try:
        parsed = ParsedQuery.model_validate(tool_block.input)
    except Exception as e:
        return {
            "parse_error": f"Schema validation failed: {e}",
            "parse_confidence": 0.0,
        }

    return {
        "cleaned_query": parsed.cleaned_query,
        "area": parsed.area,
        "price_tier": parsed.price_tier,
        "aspect_positive": parsed.aspect_positive,
        "parse_confidence": parsed.confidence,
    }


# ---------------------------------------------------------------------------
# Node 2: validate_parse
# ---------------------------------------------------------------------------
MIN_CONFIDENCE = 0.5


def validate_parse(state: AgentState) -> dict[str, Any]:
    """Pure-Python schema check on the parse output. No LLM call.

    Three rejection conditions, in order:
      1. parse_error is set (the API call or schema validation failed)
      2. parse_confidence < MIN_CONFIDENCE
      3. Any of area / price_tier / aspect_positive is set to a value
         outside the valid set (the API schema should prevent this, but
         belt-and-braces).

    Sets parse_valid for the conditional edge.
    """
    if state.parse_error:
        return {
            "parse_valid": False,
            "validation_reason": f"parse_error: {state.parse_error}",
        }

    if state.parse_confidence < MIN_CONFIDENCE:
        return {
            "parse_valid": False,
            "validation_reason": (
                f"low confidence ({state.parse_confidence:.2f} < {MIN_CONFIDENCE})"
            ),
        }

    if state.area is not None and state.area not in VALID_AREAS:
        return {
            "parse_valid": False,
            "validation_reason": f"invalid area: {state.area}",
        }
    if state.price_tier is not None and state.price_tier not in VALID_PRICE_TIERS:
        return {
            "parse_valid": False,
            "validation_reason": f"invalid price_tier: {state.price_tier}",
        }
    if state.aspect_positive is not None and state.aspect_positive not in VALID_ASPECTS:
        return {
            "parse_valid": False,
            "validation_reason": f"invalid aspect: {state.aspect_positive}",
        }

    return {
        "parse_valid": True,
        "validation_reason": "ok",
    }


# ---------------------------------------------------------------------------
# Conditional edge: route to filtered or unfiltered retrieval
# ---------------------------------------------------------------------------
def route_after_validation(state: AgentState) -> str:
    """Conditional edge function. Returns the name of the next node."""
    if state.parse_valid:
        return "retrieve_with_filters"
    return "retrieve_no_filter"


# ---------------------------------------------------------------------------
# Nodes 3a / 3b: retrieval (two paths converging on generate_answer)
# ---------------------------------------------------------------------------
def retrieve_with_filters(state: AgentState) -> dict[str, Any]:
    """Run retrieval with the parsed filters applied.

    Uses cleaned_query (the agent-stripped version) for the embedding so
    'best service in north london' becomes 'best service' before semantic
    search - more focused vector, less noise.
    """
    pipeline = _get_retrieval_pipeline()
    result = pipeline.retrieve(
        query=state.cleaned_query or state.raw_query,
        area=state.area,
        price_tier=state.price_tier,
        aspect_positive=state.aspect_positive,
    )
    return {"retrieval_result": result, "used_filters": True}


def retrieve_no_filter(state: AgentState) -> dict[str, Any]:
    """Fallback: retrieval on the raw query, no filters applied."""
    pipeline = _get_retrieval_pipeline()
    result = pipeline.retrieve(query=state.raw_query)
    return {"retrieval_result": result, "used_filters": False}


# ---------------------------------------------------------------------------
# Node 4: generate_answer
# ---------------------------------------------------------------------------
def generate_answer(state: AgentState) -> dict[str, Any]:
    """Hand the retrieval result to the existing AnswerEngine, sync (non-streaming).

    The streaming endpoint will wrap this differently (it'll bypass the
    full AnswerEngine and stream tokens directly). For the CLI test path
    and the sync /api/agent/query endpoint, this is fine.
    """
    engine = _get_answer_engine()
    retrieval = state.retrieval_result
    if retrieval is None:
        # Shouldn't happen if the graph is wired correctly
        return {"answer_result": None}

    # We re-use AnswerEngine.answer() but pass the already-computed retrieval
    # via a small bypass. Simpler: just call answer() with the same filters
    # and let it re-run retrieval. Adds ~50ms but keeps the code path simple.
    # For demo purposes the agent timing is dominated by Sonnet, not retrieval.
    if state.used_filters:
        result = engine.answer(
            query=state.cleaned_query or state.raw_query,
            area=state.area,
            price_tier=state.price_tier,
            aspect_positive=state.aspect_positive,
            stream=False,
        )
    else:
        result = engine.answer(query=state.raw_query, stream=False)

    return {"answer_result": result}