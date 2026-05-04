"""
state.py - Pydantic state model for the LangGraph query understanding agent.

The state flows through the graph: parse_query -> validate_parse -> retrieve -> generate.
Each node receives the full state and returns a partial update; LangGraph merges
the updates automatically.

State design notes:
  - We separate the user's RAW query from the agent's CLEANED query. The cleaned
    version is what gets embedded for retrieval; the raw version is what's
    logged and what we fall back to if parsing fails.
  - All filter fields are Optional. None means "no constraint", which is the
    correct default and matches what RetrievalPipeline.retrieve() accepts.
  - parse_valid is the conditional-edge signal. It defaults to None so we can
    distinguish 'not yet validated' from 'validated and failed'.
  - retrieval_result and answer are populated by the retrieval and generation
    nodes respectively. They're typed loosely (Any) to avoid a circular import
    on RetrievalResult / AnswerResult dataclasses.

Multi-turn fields:
  - prior_filters / prior_results / prior_lounge_focus carry forward from the
    previous turn's session state. None on turn 1 / fresh sessions; populated
    on turn 2+ if a session_id was supplied.
  - lounge_focus is set by the parser when the user is asking about a specific
    lounge ("tell me more about Noya"). Retrieval scopes to that lounge_id.
  - inherited_filters is the set of filter field names the parser carried over
    from prior_filters rather than freshly extracting from the new turn. Used
    by the frontend to mark pills as carried-forward.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# Closed sets for validation. Mirrored from data/lounges.csv and the ABSA
# taxonomy. If either changes, this needs updating in lockstep.
VALID_AREAS = {"North London", "Central London", "East London", "South London", "West London"}
VALID_PRICE_TIERS = {"budget", "mid", "premium"}
VALID_ASPECTS = {
    "flavour_quality", "coal_management", "service_speed", "value_for_money",
    "atmosphere_vibe", "seating_comfort", "food_quality", "wait_time",
}

# Lounge IDs are loaded from data/lounges.csv at module import time so the
# parser can validate against the canonical set. The list is small (14
# entries) and changes rarely, so an in-memory cache is fine.
def _load_lounge_ids() -> set[str]:
    """Read data/lounges.csv and return the set of valid lounge_ids.

    Done at import time so VALID_LOUNGE_IDS is available everywhere the
    state module is. Falls back to an empty set if the file isn't found
    (e.g. in unit tests run from the wrong working directory) - the parser
    will then accept any lounge_id string and validation will catch errors
    downstream.
    """
    import csv
    from pathlib import Path

    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "lounges.csv",
        Path("data/lounges.csv"),
    ]
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return {row["lounge_id"] for row in reader if row.get("lounge_id")}
    return set()


VALID_LOUNGE_IDS = _load_lounge_ids()


class ParsedQuery(BaseModel):
    """Schema returned by Haiku's extract_query_filters tool call.

    The tool input_schema is generated from this Pydantic model so any
    field added here automatically becomes part of the tool definition
    Haiku sees.

    All filter fields are Optional. None means the user did not specify
    a constraint AND no prior turn's value should be carried forward.
    See parse_query for the merge logic that distinguishes "not specified"
    from "explicitly cleared".
    """

    cleaned_query: str = Field(
        ...,
        description=(
            "The semantic core of the query for vector embedding. Strip "
            "location names and price language ONLY when they're being used "
            "as filters; keep them when they describe the experience itself."
        ),
    )
    area: Optional[str] = Field(
        None,
        description=(
            "London area filter, one of: North London, Central London, "
            "East London, South London, West London. Null if the user "
            "did not specify or imply an area."
        ),
    )
    price_tier: Optional[str] = Field(
        None,
        description=(
            "Price tier filter, one of: budget, mid, premium. Null if the "
            "user did not specify or imply a price constraint."
        ),
    )
    aspect_positive: Optional[str] = Field(
        None,
        description=(
            "Specific aspect being asked about, one of: flavour_quality, "
            "coal_management, service_speed, value_for_money, "
            "atmosphere_vibe, seating_comfort, food_quality, wait_time. "
            "Null if the query is not about a specific aspect."
        ),
    )
    lounge_focus: Optional[str] = Field(
        None,
        description=(
            "If the user is asking about a specific lounge by name "
            "(e.g. 'tell me more about Noya'), set this to the lounge_id. "
            "Null otherwise. Use the lounge_id values from the available "
            "lounges list provided in the system prompt."
        ),
    )
    is_in_taxonomy: bool = Field(
        ...,
        description=(
            "True if the query is plausibly about London shisha lounges, "
            "False if it's clearly off-topic (weather, sports, generic "
            "chat, etc). When False, the system declines without retrieval."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How confident you are in this parse, 0.0 to 1.0. Below 0.5 "
            "the system falls back to unfiltered retrieval."
        ),
    )


class AgentState(BaseModel):
    """State carried through the LangGraph workflow.

    Populated incrementally as each node runs. Multi-turn fields default
    to None / empty so single-turn behaviour is unchanged when no prior
    context is supplied.
    """

    # ---- Input (set once at graph entry) ----
    raw_query: str

    # ---- Multi-turn input (set at graph entry from session state) ----
    # All None on turn 1 / fresh sessions. Populated on turn 2+ when the
    # session_id was supplied to the endpoint.
    prior_filters: Optional[dict] = None
    prior_results: list[str] = Field(default_factory=list)

    # ---- Parse output (set by parse_query node) ----
    cleaned_query: Optional[str] = None
    area: Optional[str] = None
    price_tier: Optional[str] = None
    aspect_positive: Optional[str] = None
    lounge_focus: Optional[str] = None
    parse_confidence: float = 0.0
    is_in_taxonomy: Optional[bool] = None  # set by parse; None if parse failed
    parse_error: Optional[str] = None

    # Set of filter field names that were inherited from prior_filters
    # rather than freshly extracted from the current turn. Used by the
    # frontend to render carried-forward pills with a visual distinction.
    # Field names are strings: "area", "price_tier", "aspect_positive",
    # "lounge_focus".
    inherited_filters: set[str] = Field(default_factory=set)

    # ---- Validation output (set by validate_parse node) ----
    parse_valid: Optional[bool] = None
    validation_reason: Optional[str] = None
    is_declined: bool = False
    decline_reason: Optional[str] = None

    # ---- Retrieval output (set by retrieve_* node) ----
    retrieval_result: Optional[Any] = None
    used_filters: bool = False  # True if the with_filters branch ran

    # ---- Generation output (set by generate_answer node) ----
    answer_result: Optional[Any] = None

    class Config:
        # Allow `set[str]` to round-trip through the LangGraph state merge.
        # Pydantic v2 handles sets natively but we need arbitrary_types
        # for the loosely-typed retrieval_result / answer_result fields.
        arbitrary_types_allowed = True