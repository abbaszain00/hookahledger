"""
nodes.py - LangGraph node functions for the query understanding agent.

Five nodes:
  parse_query       - Haiku tool-use call to extract structured filters,
                      with optional merge logic against prior turn state
  validate_parse    - Pure Python schema check; sets parse_valid for routing
  retrieve_with_filters / retrieve_no_filter - Two retrieval paths
  decline_query     - Terminal node for off-taxonomy queries (no retrieval)
  generate_answer   - Reuses the existing AnswerEngine for the Sonnet call

Each node receives the current AgentState and returns a dict of fields to
update. LangGraph merges those into the state for the next node.

Multi-turn parsing:
  When state.prior_filters is None (turn 1 / fresh session), parse_query
  uses the standard prompt and Haiku extracts filters from the user's
  query in isolation.

  When state.prior_filters is populated (turn 2+ in a session), parse_query
  uses the merge prompt. Haiku is shown the prior state plus the new turn
  and decides what to inherit, replace, or clear. The set of fields that
  match the prior state (i.e. were carried forward) is computed in Python
  after the call and stored in state.inherited_filters for the frontend
  to render.

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

import csv
import json
import sys
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

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
    VALID_LOUNGE_IDS,
)


# ---------------------------------------------------------------------------
# Lazy singletons - loaded on first use, kept across requests
# ---------------------------------------------------------------------------
_anthropic_client: anthropic.Anthropic | None = None
_retrieval_pipeline: RetrievalPipeline | None = None
_answer_engine: AnswerEngine | None = None
_lounge_directory_text: str | None = None


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
        _answer_engine = AnswerEngine(retrieval_pipeline=_get_retrieval_pipeline())
    return _answer_engine


def _get_lounge_directory_text() -> str:
    """Build a compact list of available lounges for the parse prompt.

    Format: one line per lounge, "lounge_id: Lounge Name (Neighbourhood,
    Area)". Loaded once at module level and cached.

    Used in both standard and merge parse prompts so Haiku can resolve
    natural-language references like "Noya" or "the place in Seven
    Sisters" to a stable lounge_id.
    """
    global _lounge_directory_text
    if _lounge_directory_text is not None:
        return _lounge_directory_text

    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "data" / "lounges.csv",
        Path("data/lounges.csv"),
    ]
    for path in candidates:
        if path.exists():
            lines = []
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lid = row.get("lounge_id", "").strip()
                    name = row.get("lounge_name", "").strip()
                    neighbourhood = row.get("neighbourhood", "").strip()
                    area = row.get("area", "").strip()
                    if lid and name:
                        loc = f"{neighbourhood}, {area}" if neighbourhood else area
                        lines.append(f"- {lid}: {name} ({loc})")
            _lounge_directory_text = "\n".join(lines) if lines else "(none loaded)"
            return _lounge_directory_text

    _lounge_directory_text = "(none loaded)"
    return _lounge_directory_text


# ---------------------------------------------------------------------------
# Node 1: parse_query
# ---------------------------------------------------------------------------
PARSE_TOOL_DEFINITION = {
    "name": "extract_query_filters",
    "description": (
        "Extract structured filters from a natural-language query about "
        "London shisha lounges. Identify the area, price tier, main aspect "
        "being asked about, and whether the user is focused on a specific "
        "lounge. Return a cleaned semantic query for vector retrieval."
    ),
    "input_schema": ParsedQuery.model_json_schema(),
}


# Prompt used on turn 1 (no prior turn state). Single-turn extraction.
STANDARD_PARSE_SYSTEM_PROMPT = """You are a query understanding assistant for a London shisha lounge search system. Your job is to read a natural-language query and extract structured filters.

The query may contain:
- An area or neighbourhood (Soho, Edgware Road, Kingsbury, etc)
- A price constraint (under £20, premium, cheap, etc)
- A specific aspect being asked about (good service, smooth shisha, food quality, etc)
- A specific lounge by name (Noya, Tigerbay, etc)
- Or none of these - a general "best lounge" query

Price tier mapping:
- budget: typical spend ~£15/head. Use for "cheap", "very affordable", "under £18", "under £20".
- mid: typical spend ~£20-25/head. Use for "mid-range", "moderate", "reasonably priced", "under £25", "under £30", "around £25".
- premium: typical spend £30+/head. Use for "premium", "upscale", "high-end", "splurge", "fancy".

When the user gives a numeric ceiling, pick the tier whose typical spend is at or just under the ceiling, not the cheapest tier that fits. "Under £25" is mid (mid ~£20-25 satisfies "under £25"; budget ~£15 satisfies it too but is unnecessarily restrictive). "Under £20" is budget. "Under £30" is mid.

You must also classify whether the query is about shisha lounges at all. Set is_in_taxonomy=False ONLY for queries that are clearly off-topic (weather, sports, news, generic chat, coding help, unrelated cuisines, etc). Set is_in_taxonomy=True for anything that could plausibly be answered from shisha lounge reviews, even if it's vague or you can't extract structured filters from it.

For lounge_focus: if the user is asking about a specific named lounge, set this to the lounge_id from the available lounges list below. Use the FULL canonical lounge_id verbatim - do NOT shorten or simplify it.

Examples:
- "Noya" or "tell me about Noya" -> "noya_harringay" (NOT "noya")
- "the Banc" -> "the_banc_seven_sisters" (NOT "the_banc" or "banc")
- "Tigerbay" -> "tigerbay_kingsbury" (NOT "tigerbay")
- "Al-Dar" -> "aldar_edgware" (NOT "aldar" or "al-dar")

If the user's reference is ambiguous or doesn't match any lounge in the list below, leave lounge_focus null. Otherwise leave it null.
Available lounges:
{LOUNGE_DIRECTORY}

Use the extract_query_filters tool to return your parse. Always call the tool exactly once.

Be conservative on filters: if you're not sure whether the query implies a filter, return null for that field rather than guessing. Confidence below 0.5 means the query is too vague to filter on - the system will fall back to unfiltered retrieval."""


# Prompt used on turn 2+ when there's prior turn state. The job is now to
# MERGE the new turn with the prior state, deciding what to inherit, what
# to replace, and what to clear.
MERGE_PARSE_SYSTEM_PROMPT = """You are a query understanding assistant for a London shisha lounge search system handling a multi-turn conversation. The user has been searching across multiple turns; you must merge the new turn with the prior turn's state.

The user's prior turn established this filter state:
{PRIOR_STATE}

The lounges that surfaced in the prior turn's results (in order):
{PRIOR_RESULTS}

The new turn from the user:
"{RAW_QUERY}"

Your job is to decide what the user means in context, and produce a MERGED filter state that reflects their intent. Three operations apply:

1. INHERIT: if the new turn doesn't address a filter that was set previously, carry it forward. Example: prior had area=North London; new turn says "under £25" - keep area=North London, set price_tier=budget.

2. REPLACE: if the new turn explicitly mentions a different value for a filter, replace it. Example: prior had area=North London; new turn says "actually let's try central london" - replace area with Central London.

3. CLEAR: if the new turn shifts scope in a way that makes prior context no longer apply, clear stale fields. Example: changing area should clear lounge_focus (the focused lounge is no longer in scope). Asking a fresh broad question ("what about something completely different") should clear most filters.

Specific guidance:

- If the new turn is a follow-up about a specific lounge (e.g. "tell me more about Noya", "what about food there", "is it any good for service"), set lounge_focus to the relevant lounge_id. "There" or "it" or "the first one" refers to a lounge from the prior results - resolve based on context, defaulting to the most recently focused or the top result.

- When lounge_focus is set, area and price_tier filters are usually not needed (we're already scoped to one lounge). Either inherit them quietly or null them - it doesn't change retrieval much.

- If the new turn introduces a new area, clear any existing lounge_focus (the focused lounge may not be in the new area).

- If the new turn is a fresh broad question that doesn't reference prior context (e.g. "actually I want to find somewhere chilled"), reset most filters and treat it nearly as turn 1.

- Aspect inheritance: aspects often DON'T inherit. "best atmosphere in north london" -> "under £25" - the user is still asking about atmosphere lounges, just with an added price constraint, so aspect=atmosphere_vibe still applies. But "best atmosphere in north london" -> "what about food there" - the user has switched aspects from atmosphere to food. Use linguistic cues; when in doubt, infer the new aspect from the new turn rather than inheriting.

- Price tier mapping when the user gives a numeric ceiling: pick the tier whose typical spend is at or just under the ceiling, not the cheapest tier that fits. budget = ~£15/head, mid = ~£20-25/head, premium = £30+/head. "Under £20" -> budget. "Under £25" -> mid (mid satisfies the constraint without being unnecessarily restrictive). "Under £30" -> mid. "Under £35" -> premium boundary; usually mid still works.
You must also classify whether the new turn is on-topic. Set is_in_taxonomy=False ONLY for queries that are clearly off-topic (weather, sports, generic chat). Set is_in_taxonomy=True for any plausible follow-up about lounges, even vague or terse ones like "and food?" or "the first one?".

For lounge_focus: match natural-language references to the canonical id from the available lounges list. Use the FULL canonical lounge_id verbatim - do NOT shorten or simplify it.

Examples:
- "Noya" or "tell me about Noya" -> "noya_harringay" (NOT "noya")
- "Tigerbay" -> "tigerbay_kingsbury" (NOT "tigerbay")
- "the Banc" -> "the_banc_seven_sisters" (NOT "the_banc" or "banc")

If the reference is ambiguous, leave lounge_focus null and let the system fall back to broader retrieval.
Available lounges:
{LOUNGE_DIRECTORY}

Use the extract_query_filters tool to return your merged parse. Always call the tool exactly once.

Confidence below 0.5 means the merge was too uncertain - the system will fall back to unfiltered retrieval."""


def _format_prior_state(prior_filters: dict | None) -> str:
    """Compact human-readable rendering of prior filter state for the merge prompt."""
    if not prior_filters:
        return "(no prior filters)"
    parts = []
    for key in ("area", "price_tier", "aspect_positive", "lounge_focus"):
        val = prior_filters.get(key)
        parts.append(f"  {key}: {val if val is not None else 'null'}")
    return "\n".join(parts)


def _format_prior_results(prior_results: list[str]) -> str:
    """Compact rendering of the prior turn's lounge order for the merge prompt."""
    if not prior_results:
        return "(no prior results)"
    return "\n".join(f"  {i + 1}. {lid}" for i, lid in enumerate(prior_results[:10]))


def _compute_inherited_filters(parsed: ParsedQuery, prior_filters: dict | None) -> set[str]:
    """Identify which filter fields in the parsed result match prior_filters.

    A field is "inherited" iff it has a non-null value AND that value is
    identical to the prior turn's value. Fields that were null in prior
    and are null now are not "inherited" - they were never set.

    Used to render the inferred-filters pill row with carried-forward
    indicators.
    """
    if not prior_filters:
        return set()

    inherited = set()
    for key in ("area", "price_tier", "aspect_positive", "lounge_focus"):
        new_val = getattr(parsed, key)
        old_val = prior_filters.get(key)
        if new_val is not None and new_val == old_val:
            inherited.add(key)
    return inherited


def parse_query(state: AgentState) -> dict[str, Any]:
    """Call Haiku with the extract_query_filters tool to parse the user's query.

    Branches on whether prior_filters is set:
      - None: standard single-turn extraction
      - Set: merge against prior state, computing inherited_filters

    Returns a state update with the parsed fields, or sets parse_error if
    the call failed. Does NOT raise - failure is a routable state, not an
    exception.
    """
    client = _get_anthropic_client()
    lounge_dir = _get_lounge_directory_text()

    # Build the system prompt based on whether we have prior state
    if state.prior_filters:
        system = (
            MERGE_PARSE_SYSTEM_PROMPT
            .replace("{PRIOR_STATE}", _format_prior_state(state.prior_filters))
            .replace("{PRIOR_RESULTS}", _format_prior_results(state.prior_results))
            .replace("{RAW_QUERY}", state.raw_query)
            .replace("{LOUNGE_DIRECTORY}", lounge_dir)
        )
    else:
        system = STANDARD_PARSE_SYSTEM_PROMPT.replace("{LOUNGE_DIRECTORY}", lounge_dir)

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=600,
            temperature=0,
            system=system,
            tools=[PARSE_TOOL_DEFINITION],
            tool_choice={"type": "tool", "name": "extract_query_filters"},
            messages=[{"role": "user", "content": state.raw_query}],
        )
    except (anthropic.APIError, ValueError) as e:
        # API error - log it, fall back to unfiltered. Validation node will
        # see parse_error set and route accordingly. Tightened from bare
        # Exception per the Day 4 audit.
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

    try:
        parsed = ParsedQuery.model_validate(tool_block.input)
    except ValidationError as e:
        return {
            "parse_error": f"Schema validation failed: {e}",
            "parse_confidence": 0.0,
        }

    # Compute which filters were inherited from prior state, if any.
    inherited = _compute_inherited_filters(parsed, state.prior_filters)

    return {
        "cleaned_query": parsed.cleaned_query,
        "area": parsed.area,
        "price_tier": parsed.price_tier,
        "aspect_positive": parsed.aspect_positive,
        "lounge_focus": parsed.lounge_focus,
        "parse_confidence": parsed.confidence,
        "is_in_taxonomy": parsed.is_in_taxonomy,
        "inherited_filters": inherited,
    }

# ---------------------------------------------------------------------------
# Node 2: validate_parse
# ---------------------------------------------------------------------------
MIN_CONFIDENCE = 0.5


def _normalise_lounge_id(candidate: str) -> str | None:
    """Try to match a possibly-shortened lounge id against VALID_LOUNGE_IDS.

    Strategies, in order:
    1. Case-insensitive exact match
    2. Prefix match: candidate is a prefix of exactly one canonical id
    3. Substring match: candidate appears in exactly one canonical id

    Returns the canonical id on unambiguous match, None otherwise. We
    require uniqueness because guessing the wrong lounge would silently
    misroute the user's query - better to fail and fall back to unfiltered
    retrieval than to scope to the wrong lounge.
    """
    if not candidate:
        return None
    cand_lower = candidate.lower().strip()

    # Strategy 1: case-insensitive exact match
    for lid in VALID_LOUNGE_IDS:
        if lid.lower() == cand_lower:
            return lid

    # Strategy 2: prefix match (e.g. "noya" -> "noya_harringay")
    prefix_matches = [
        lid for lid in VALID_LOUNGE_IDS
        if lid.lower().startswith(cand_lower + "_")
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    # Strategy 3: substring match (e.g. "harringay" -> "noya_harringay")
    substring_matches = [
        lid for lid in VALID_LOUNGE_IDS
        if cand_lower in lid.lower()
    ]
    if len(substring_matches) == 1:
        return substring_matches[0]

    return None


def validate_parse(state: AgentState) -> dict[str, Any]:
    """Pure-Python schema check on the parse output. No LLM call.

    Rejection conditions, in order:
      1. parse_error is set (the API call or schema validation failed)
      2. parse_confidence < MIN_CONFIDENCE
      3. Any of area / price_tier / aspect_positive is set to a value
         outside the valid set
      4. lounge_focus is set but cannot be matched (even tolerantly) to
         a canonical lounge_id

    Sets parse_valid for the conditional edge. Also normalises lounge_focus
    when Haiku returned a shortened form (e.g. "noya" -> "noya_harringay").
    """
    if state.parse_error:
        return {
            "parse_valid": False,
            "validation_reason": f"parse_failed: {state.parse_error}",
        }

    if state.parse_confidence < MIN_CONFIDENCE:
        return {
            "parse_valid": False,
            "validation_reason": f"low_confidence ({state.parse_confidence:.2f})",
        }

    if state.area and state.area not in VALID_AREAS:
        return {
            "parse_valid": False,
            "validation_reason": f"invalid_area: {state.area}",
        }
    if state.price_tier and state.price_tier not in VALID_PRICE_TIERS:
        return {
            "parse_valid": False,
            "validation_reason": f"invalid_price_tier: {state.price_tier}",
        }
    if state.aspect_positive and state.aspect_positive not in VALID_ASPECTS:
        return {
            "parse_valid": False,
            "validation_reason": f"invalid_aspect: {state.aspect_positive}",
        }

    # Validate (and tolerantly normalise) lounge_focus against VALID_LOUNGE_IDS.
    # Haiku occasionally returns shortened forms; attempt a tolerant match
    # before failing. If we successfully normalise, return the canonical id
    # so LangGraph propagates it to downstream nodes.
    normalised_lounge_focus = state.lounge_focus
    if state.lounge_focus and VALID_LOUNGE_IDS:
        if state.lounge_focus not in VALID_LOUNGE_IDS:
            match = _normalise_lounge_id(state.lounge_focus)
            if match is None:
                return {
                    "parse_valid": False,
                    "validation_reason": f"invalid_lounge_focus: {state.lounge_focus}",
                }
            normalised_lounge_focus = match

    return {
        "parse_valid": True,
        "validation_reason": "ok",
        "lounge_focus": normalised_lounge_focus,
    }




# ---------------------------------------------------------------------------
# Node 3a: retrieve_with_filters
# ---------------------------------------------------------------------------
def retrieve_with_filters(state: AgentState) -> dict[str, Any]:
    """Run retrieval with the parsed filters. Selected by the conditional
    edge when parse_valid is True."""
    pipeline = _get_retrieval_pipeline()
    result = pipeline.retrieve(
        query=state.cleaned_query or state.raw_query,
        area=state.area,
        price_tier=state.price_tier,
        aspect_positive=state.aspect_positive,
        lounge_id_focus=state.lounge_focus,
    )
    return {"retrieval_result": result, "used_filters": True}


def retrieve_no_filter(state: AgentState) -> dict[str, Any]:
    """Fallback: retrieval on the raw query, no filters applied."""
    pipeline = _get_retrieval_pipeline()
    result = pipeline.retrieve(query=state.raw_query)
    return {"retrieval_result": result, "used_filters": False}


# ---------------------------------------------------------------------------
# Node 3b: decline_query (terminal)
# ---------------------------------------------------------------------------
DECLINE_MESSAGE = (
    "I can only answer questions about London shisha lounges - their "
    "service, atmosphere, flavour and coal management, prices, food, "
    "seating, wait times, and locations across North, Central, East, "
    "South and West London. Try asking about lounges, areas, or what "
    "experience you're looking for."
)


def decline_query(state: AgentState) -> dict[str, Any]:
    """Terminal node for queries flagged as out-of-taxonomy by the parser.

    Returns a synthetic 'answer' with the decline message, no retrieval,
    no Sonnet call. The streaming endpoint emits this as a single token
    event so the existing answer-card render path works.
    """
    # Build a minimal answer-result-shaped object with just the text field.
    # We don't import AnswerResult to avoid a circular dependency; the
    # streaming endpoint reads .text and ignores everything else on the
    # decline path.
    class _DeclineAnswer:
        def __init__(self, text):
            self.text = text
            self.text_validated = text
            self.quote_validations = []
            self.tokens_in = 0
            self.tokens_out = 0
            self.cost_usd = 0.0

    return {
        "answer_result": _DeclineAnswer(DECLINE_MESSAGE),
        "is_declined": True,
        "decline_reason": "out_of_taxonomy",
    }


# ---------------------------------------------------------------------------
# Node 4: generate_answer
# ---------------------------------------------------------------------------
def generate_answer(state: AgentState) -> dict[str, Any]:
    """Hand the retrieval result to the existing AnswerEngine.

    Used by the sync /api/agent/query endpoint and the CLI test path.
    The streaming endpoint bypasses this and calls Sonnet directly so it
    can stream tokens.
    """
    engine = _get_answer_engine()
    retrieval = state.retrieval_result
    if retrieval is None:
        return {"answer_result": None}

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


# ---------------------------------------------------------------------------
# Conditional edge after validate_parse
# ---------------------------------------------------------------------------
def route_after_validation(state: AgentState) -> str:
    """Conditional edge selector returning the name of the next node."""
    if state.is_in_taxonomy is False:
        return "decline_query"
    if state.parse_valid:
        return "retrieve_with_filters"
    return "retrieve_no_filter"