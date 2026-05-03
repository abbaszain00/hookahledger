"""
graph.py - the LangGraph workflow definition.

Topology:

  START
    |
    v
  parse_query --> validate_parse --> [conditional]
                                       |
                  +--------------------+--------------------+
                  | parse_valid==True                       | parse_valid==False
                  v                                         v
            retrieve_with_filters                   retrieve_no_filter
                  |                                         |
                  +-------------------+---------------------+
                                      |
                                      v
                                generate_answer
                                      |
                                      v
                                     END

The conditional edge after validate_parse is the part that justifies using
LangGraph rather than a flat function chain. If parsing succeeds, we run
filtered retrieval; if it fails (low confidence, schema violation, API
error), we fall back to no-filter retrieval. Both paths converge on
generate_answer.

Compiled graph is cached at module level - the graph itself is stateless,
each invocation gets its own state instance.
"""

from __future__ import annotations

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import (
    parse_query,
    validate_parse,
    retrieve_with_filters,
    retrieve_no_filter,
    generate_answer,
    decline_query,
    route_after_validation,
)


def build_graph():
    """Construct and compile the agent graph.

    Returns a compiled CompiledStateGraph that exposes .invoke(state) and
    .stream(state) for sync and async execution respectively.
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("parse_query", parse_query)
    workflow.add_node("validate_parse", validate_parse)
    workflow.add_node("retrieve_with_filters", retrieve_with_filters)
    workflow.add_node("retrieve_no_filter", retrieve_no_filter)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("decline_query", decline_query)

    # Linear edges: START -> parse_query -> validate_parse
    workflow.add_edge(START, "parse_query")
    workflow.add_edge("parse_query", "validate_parse")

    # Conditional edge: validate_parse routes to one of three nodes
    workflow.add_conditional_edges(
        "validate_parse",
        route_after_validation,
        {
            "retrieve_with_filters": "retrieve_with_filters",
            "retrieve_no_filter": "retrieve_no_filter",
            "decline_query": "decline_query",
        },
    )

    # Both retrieval branches converge on generate_answer
    workflow.add_edge("retrieve_with_filters", "generate_answer")
    workflow.add_edge("retrieve_no_filter", "generate_answer")
    workflow.add_edge("generate_answer", END)
    # Decline is terminal - no retrieval, no generation
    workflow.add_edge("decline_query", END)

    return workflow.compile()


# Module-level cached graph. Built once per process.
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph