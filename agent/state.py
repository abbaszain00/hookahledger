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


class AgentState(BaseModel):
    """State carried through the LangGraph workflow.

    Populated incrementally as each node runs.
    """

    # ---- Input (set once at graph entry) ----
    raw_query: str

    # ---- Parse output (set by parse_query node) ----
    cleaned_query: Optional[str] = None
    area: Optional[str] = None
    price_tier: Optional[str] = None
    aspect_positive: Optional[str] = None
    parse_confidence: float = 0.0
    parse_error: Optional[str] = None  # populated if the LLM call failed

    # ---- Validation output (set by validate_parse node) ----
    parse_valid: Optional[bool] = None
    validation_reason: Optional[str] = None

    # ---- Retrieval output (set by retrieve_* node) ----
    # Loose type to avoid circular imports. Holds a RetrievalResult dataclass.
    retrieval_result: Optional[Any] = None
    used_filters: bool = False  # True if the with_filters branch ran

    # ---- Generation output (set by generate_answer node) ----
    # Loose type. Holds an AnswerResult dataclass.
    answer_result: Optional[Any] = None

    # Allow holding arbitrary dataclasses in retrieval_result / answer_result
    model_config = {"arbitrary_types_allowed": True}


class ParsedQuery(BaseModel):
    """Schema for the structured output Haiku returns from the parse_query
    tool call. Kept separate from AgentState so it maps cleanly to the
    Anthropic tool input_schema."""

    cleaned_query: str = Field(
        ...,
        description=(
            "The user's query with location and constraint phrases stripped, "
            "so it can be used for semantic retrieval. E.g. 'best service in "
            "north london under 25' becomes 'best service'. Keep adjectives "
            "describing the desired experience (smooth, premium, etc) but "
            "remove location names, prices, and area constraints."
        ),
    )
    area: Optional[str] = Field(
        None,
        description=(
            "London area mentioned in the query, if any. Must be one of: "
            "'North London', 'Central London', 'East London', 'South London', "
            "'West London'. Map common neighbourhoods correctly: Soho/Mayfair/"
            "Edgware Road = Central; Kingsbury/Harringay/Wood Green/Edgware/"
            "Seven Sisters/Turnpike Lane = North; Forest Gate = East; "
            "Streatham/Clapham/Vauxhall/Brixton = South. Return null if no "
            "area is mentioned or implied."
        ),
    )
    price_tier: Optional[str] = Field(
        None,
        description=(
            "Price tier mentioned in the query, if any. Must be 'budget' "
            "(under ~£18), 'mid' (£18-28), or 'premium' (£28+). Map specific "
            "prices: 'under £20' = budget, 'around £25' or 'under £25' = mid, "
            "'over £30' or 'expensive' or 'high-end' = premium. Return null "
            "if no price is mentioned."
        ),
    )
    aspect_positive: Optional[str] = Field(
        None,
        description=(
            "If the user is asking about a specific quality, return the "
            "matching aspect. Must be one of: 'flavour_quality' (taste, "
            "flavours), 'coal_management' (smooth, lasts, doesn't run out), "
            "'service_speed' (staff, attentive), 'value_for_money' (cheap, "
            "good value), 'atmosphere_vibe' (vibe, decor, music), "
            "'seating_comfort' (seats, space), 'food_quality' (food, dishes), "
            "'wait_time' (queue, booking). Return null for general queries "
            "like 'best lounge' that don't focus on a single aspect."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How confident you are this parse is correct. 0.9+ for clear "
            "queries with explicit area/aspect signals. 0.5-0.8 for ambiguous "
            "queries where you inferred filters from context. Below 0.5 if "
            "the query is too vague to parse reliably."
        ),
    )