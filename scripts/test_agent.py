"""
test_agent.py - exercise the agent graph end-to-end from the CLI.

Runs a query through the full pipeline (parse -> validate -> retrieve -> generate)
and prints what each node produced. Useful for:
  - verifying the agent works before plumbing it into FastAPI
  - eyeballing parse quality on different query types
  - debugging the conditional routing

Usage:
  python scripts/test_agent.py "best service in north london under 25"
  python scripts/test_agent.py "smooth shisha that lasts" --verbose
  python scripts/test_agent.py "what's the meaning of life"   # tests fallback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.graph import get_graph
from agent.state import AgentState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Natural language query to parse and answer")
    parser.add_argument("--verbose", action="store_true",
                        help="Print intermediate state at each node")
    args = parser.parse_args()

    graph = get_graph()
    initial = AgentState(raw_query=args.query)

    print(f"Query: \"{args.query}\"")
    print("-" * 70)

    if args.verbose:
        # stream() yields a dict per node as it executes - {node_name: state_update}
        # Useful for seeing exactly what each node produces.
        final_state = None
        for step in graph.stream(initial, stream_mode="values"):
            final_state = step
            # stream_mode="values" yields the FULL state after each node, not the diff
        # Replay with stream_mode="updates" for per-node visibility
        print("\n[Per-node updates]\n")
        for step in graph.stream(initial, stream_mode="updates"):
            for node_name, update in step.items():
                print(f"--- {node_name} ---")
                for k, v in (update or {}).items():
                    if k in ("retrieval_result", "answer_result"):
                        # Don't dump the full dataclass; summarise
                        if v is None:
                            print(f"  {k}: None")
                        elif k == "retrieval_result":
                            n_chunks = len(getattr(v, "chunks", []))
                            n_lounges = len(getattr(v, "lounges", []))
                            print(f"  {k}: {n_chunks} chunks across {n_lounges} lounges")
                        elif k == "answer_result":
                            print(f"  {k}: <AnswerResult, {len(v.text)} chars, "
                                  f"${v.cost_usd:.4f}>")
                    else:
                        print(f"  {k}: {v}")
                print()
    else:
        final_state = graph.invoke(initial)

    if final_state is None:
        print("ERROR: no final state")
        sys.exit(1)

    # In stream_mode="values", final_state is the full AgentState dict at the end.
    # In invoke(), final_state is also the full state. Same shape either way.
    print("\n" + "=" * 70)
    print("FINAL")
    print("=" * 70)
    print(f"Cleaned query:     {final_state.get('cleaned_query')}")
    print(f"Area:              {final_state.get('area')}")
    print(f"Price tier:        {final_state.get('price_tier')}")
    print(f"Aspect:            {final_state.get('aspect_positive')}")
    print(f"Parse confidence:  {final_state.get('parse_confidence'):.2f}")
    print(f"Parse valid:       {final_state.get('parse_valid')}")
    print(f"Validation reason: {final_state.get('validation_reason')}")
    print(f"Used filters:      {final_state.get('used_filters')}")

    answer = final_state.get("answer_result")
    if answer is not None:
        print()
        print("ANSWER:")
        print(answer.text_validated)
        print()
        print(f"[tokens: {answer.tokens_in} in / {answer.tokens_out} out · "
              f"cost: ${answer.cost_usd:.4f}]")


if __name__ == "__main__":
    main()