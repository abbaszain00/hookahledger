"""
run_eval.py - score retrieval precision@5 against the hand-curated eval set.

For each query in tests/eval_queries.json, runs up to FOUR passes:
  1. NO RERANK            - similarity * recency only, JSON filters
  2. RERANK               - rerank-v3.5 over the bare query, JSON filters
  3. FILTERED+RERANK      - rerank with aspect_positive filter applied
                            (only when query has an aspect_filter field)
  4. AGENT+RERANK         - LangGraph parse_query infers the filters from
                            natural language; rerank applied. Uses the
                            agent's filters, not the JSON's.

Each pass is scored precision@5: 1 if any expected_lounges appears in the
top-5 surfaced lounges, else 0.

The agent pass is the headline experiment. If precision is similar to or
better than the JSON-filter pass, that means the agent can derive the same
filters the human picked - which is the demo claim. If it underperforms,
that's writeup material.

Cohere trial keys are limited to 10 rerank calls per minute. The runner
sleeps 6.5s between rerank calls. With this v4 set (~30 rerank calls)
that adds ~3 min of wait time but produces honest numbers instead of
triggering the Fix 2 fallback path.

Usage:
  python scripts/run_eval.py
  python scripts/run_eval.py --no-sleep        # paid Cohere key
  python scripts/run_eval.py --skip-agent      # v3 behaviour
  python scripts/run_eval.py --json > eval_results.json
"""

from __future__ import annotations

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieve import RetrievalPipeline  # noqa: E402

# Agent imports - only used if --skip-agent is not set
from agent.nodes import parse_query, validate_parse  # noqa: E402
from agent.state import AgentState  # noqa: E402


RERANK_SLEEP_SECONDS = 6.5


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class QueryRun:
    pass_label: str
    surfaced_lounges: list[str]
    passed: bool
    n_chunks: int
    # Agent pass only - what the agent inferred
    agent_filters: dict | None = None


@dataclass
class QueryResult:
    id: str
    category: str
    query: str
    filters: dict
    aspect_filter: str | None
    expected_lounges: list[str]
    runs: list[QueryRun]


@dataclass
class PassMetrics:
    n_eligible: int
    n_passed: int
    precision_at_k: float


@dataclass
class EvalSummary:
    n_queries: int
    no_rerank: PassMetrics
    rerank: PassMetrics
    filtered_rerank: PassMetrics
    agent_rerank: PassMetrics
    rerank_lift_points: float
    filter_lift_points: float
    agent_lift_points: float
    by_category: dict


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_top_k(surfaced: list[str], expected: list[str]) -> bool:
    return any(lounge in surfaced for lounge in expected)


def run_one_pass(
    pipeline: RetrievalPipeline,
    query: str,
    filters: dict,
    aspect_positive: str | None,
    expected_lounges: list[str],
    top_k: int,
    rerank: bool,
    pass_label: str,
    agent_filters: dict | None = None,
) -> QueryRun:
    result = pipeline.retrieve(
        query=query,
        area=filters.get("area"),
        price_tier=filters.get("price_tier"),
        aspect_positive=aspect_positive,
        top_k=top_k,
        rerank=rerank,
    )
    seen: list[str] = []
    for chunk in result.chunks:
        if chunk.lounge_id not in seen:
            seen.append(chunk.lounge_id)
    return QueryRun(
        pass_label=pass_label,
        surfaced_lounges=seen,
        passed=score_top_k(seen, expected_lounges),
        n_chunks=len(result.chunks),
        agent_filters=agent_filters,
    )


def run_agent_parse(query: str) -> dict:
    """Run only the parse_query + validate_parse nodes from the graph.

    Returns a dict {area, price_tier, aspect_positive, cleaned_query, parse_valid}.
    Used in the agent pass to derive filters from natural language without
    invoking the full graph (no Sonnet call needed - we only want the
    retrieval result).
    """
    state = AgentState(raw_query=query)
    parse_update = parse_query(state)
    for k, v in parse_update.items():
        setattr(state, k, v)
    validate_update = validate_parse(state)
    for k, v in validate_update.items():
        setattr(state, k, v)
    return {
        "area": state.area if state.parse_valid else None,
        "price_tier": state.price_tier if state.parse_valid else None,
        "aspect_positive": state.aspect_positive if state.parse_valid else None,
        "cleaned_query": state.cleaned_query if state.parse_valid else query,
        "parse_valid": state.parse_valid,
        "parse_confidence": state.parse_confidence,
    }


def run_eval(
    pipeline: RetrievalPipeline,
    queries: list[dict],
    top_k: int = 5,
    sleep_between_rerank: float = RERANK_SLEEP_SECONDS,
    skip_agent: bool = False,
) -> tuple[list[QueryResult], EvalSummary]:
    results: list[QueryResult] = []
    rerank_calls_so_far = 0

    def _maybe_sleep_for_cohere() -> None:
        nonlocal rerank_calls_so_far
        if sleep_between_rerank > 0 and rerank_calls_so_far > 0:
            time.sleep(sleep_between_rerank)
        rerank_calls_so_far += 1

    total_queries = len(queries)
    for idx, q in enumerate(queries, 1):
        filters = q.get("filters", {})
        aspect_filter = q.get("aspect_filter")
        expected = q["expected_lounges"]

        print(f"[{idx}/{total_queries}] {q['id']}", file=sys.stderr)

        # Pass 1: no rerank, JSON filters, no aspect filter
        no_rerank_run = run_one_pass(
            pipeline, q["query"], filters, None, expected,
            top_k, rerank=False, pass_label="no_rerank",
        )

        # Pass 2: rerank, JSON filters, no aspect filter
        _maybe_sleep_for_cohere()
        rerank_run = run_one_pass(
            pipeline, q["query"], filters, None, expected,
            top_k, rerank=True, pass_label="rerank",
        )
        runs = [no_rerank_run, rerank_run]

        # Pass 3: rerank + JSON aspect filter
        if aspect_filter:
            _maybe_sleep_for_cohere()
            filtered_run = run_one_pass(
                pipeline, q["query"], filters, aspect_filter, expected,
                top_k, rerank=True, pass_label="filtered_rerank",
            )
            runs.append(filtered_run)

        # Pass 4: agent-inferred filters + rerank
        if not skip_agent:
            print(f"  ... running agent parse", file=sys.stderr)
            agent_parse = run_agent_parse(q["query"])
            agent_filters_dict = {
                "area": agent_parse["area"],
                "price_tier": agent_parse["price_tier"],
            }
            _maybe_sleep_for_cohere()
            agent_run = run_one_pass(
                pipeline,
                # Use cleaned_query if parse was valid, else raw query
                agent_parse["cleaned_query"],
                agent_filters_dict,
                agent_parse["aspect_positive"],
                expected,
                top_k,
                rerank=True,
                pass_label="agent_rerank",
                agent_filters={
                    "area": agent_parse["area"],
                    "price_tier": agent_parse["price_tier"],
                    "aspect_positive": agent_parse["aspect_positive"],
                    "cleaned_query": agent_parse["cleaned_query"],
                    "parse_valid": agent_parse["parse_valid"],
                    "confidence": agent_parse["parse_confidence"],
                },
            )
            runs.append(agent_run)

        results.append(QueryResult(
            id=q["id"],
            category=q["category"],
            query=q["query"],
            filters=filters,
            aspect_filter=aspect_filter,
            expected_lounges=expected,
            runs=runs,
        ))

    # ---- Aggregate ----
    n_queries = len(results)

    def _pass_metrics(label: str) -> PassMetrics:
        runs = [r for q in results for r in q.runs if r.pass_label == label]
        n_eligible = len(runs)
        n_passed = sum(1 for r in runs if r.passed)
        precision = n_passed / n_eligible if n_eligible else 0.0
        return PassMetrics(
            n_eligible=n_eligible, n_passed=n_passed, precision_at_k=precision,
        )

    no_rerank_m = _pass_metrics("no_rerank")
    rerank_m = _pass_metrics("rerank")
    filtered_m = _pass_metrics("filtered_rerank")
    agent_m = _pass_metrics("agent_rerank")

    # filter_lift: only fair to compute over queries that HAVE a filter.
    filter_eligible_ids = {
        q.id for q in results
        if any(r.pass_label == "filtered_rerank" for r in q.runs)
    }
    rerank_on_filterable = [
        r for q in results for r in q.runs
        if r.pass_label == "rerank" and q.id in filter_eligible_ids
    ]
    rerank_baseline_for_filter = (
        sum(1 for r in rerank_on_filterable if r.passed) / len(rerank_on_filterable)
        if rerank_on_filterable else 0.0
    )
    filter_lift = (filtered_m.precision_at_k - rerank_baseline_for_filter) * 100.0

    # agent_lift: agent vs vanilla rerank baseline (all queries)
    agent_lift = (agent_m.precision_at_k - rerank_m.precision_at_k) * 100.0

    by_cat: dict = {}
    for r in results:
        cat = r.category
        slot = by_cat.setdefault(cat, {
            "n": 0,
            "passed_no_rerank": 0,
            "passed_rerank": 0,
            "passed_filtered_rerank": 0,
            "passed_agent_rerank": 0,
            "n_filterable": 0,
            "n_agent": 0,
        })
        slot["n"] += 1
        for run in r.runs:
            if run.pass_label == "no_rerank" and run.passed:
                slot["passed_no_rerank"] += 1
            elif run.pass_label == "rerank" and run.passed:
                slot["passed_rerank"] += 1
            elif run.pass_label == "filtered_rerank":
                slot["n_filterable"] += 1
                if run.passed:
                    slot["passed_filtered_rerank"] += 1
            elif run.pass_label == "agent_rerank":
                slot["n_agent"] += 1
                if run.passed:
                    slot["passed_agent_rerank"] += 1

    summary = EvalSummary(
        n_queries=n_queries,
        no_rerank=no_rerank_m,
        rerank=rerank_m,
        filtered_rerank=filtered_m,
        agent_rerank=agent_m,
        rerank_lift_points=(rerank_m.precision_at_k - no_rerank_m.precision_at_k) * 100.0,
        filter_lift_points=filter_lift,
        agent_lift_points=agent_lift,
        by_category=by_cat,
    )
    return results, summary


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_human(results: list[QueryResult], summary: EvalSummary) -> None:
    print("=" * 90)
    print("HOOKAHLEDGER EVAL: precision@5 retrieval scoring")
    print("=" * 90)
    print()

    for r in results:
        runs_by_label = {run.pass_label: run for run in r.runs}

        filter_str = ""
        if r.filters:
            filter_str = " | filters: " + ", ".join(
                f"{k}={v}" for k, v in r.filters.items()
            )
        if r.aspect_filter:
            filter_str += f" | aspect_filter: {r.aspect_filter}"

        print(f"--- {r.id} ({r.category}) ---")
        print(f"  Query:    \"{r.query}\"{filter_str}")
        print(f"  Expected: {', '.join(r.expected_lounges)}")

        for label in ("no_rerank", "rerank", "filtered_rerank", "agent_rerank"):
            if label not in runs_by_label:
                continue
            run = runs_by_label[label]
            marker = "[PASS]" if run.passed else "[FAIL]"
            display = label.replace("_", " ").ljust(16)
            surfaced = ", ".join(run.surfaced_lounges) or "(empty)"
            print(f"  {display} {marker}: {surfaced}")
            if label == "agent_rerank" and run.agent_filters:
                af = run.agent_filters
                inferred = ", ".join(
                    f"{k}={v}" for k, v in {
                        "area": af.get("area"),
                        "price_tier": af.get("price_tier"),
                        "aspect_positive": af.get("aspect_positive"),
                    }.items() if v
                ) or "(none inferred)"
                print(f"                     ↳ agent inferred: {inferred} "
                      f"(conf {af.get('confidence', 0):.2f})")
        print()

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    nr = summary.no_rerank
    rr = summary.rerank
    fr = summary.filtered_rerank
    ar = summary.agent_rerank
    print(f"  Queries:                       {summary.n_queries}")
    print(f"  Pass 1: no rerank              {nr.n_passed}/{nr.n_eligible}  "
          f"= {nr.precision_at_k:.1%}")
    print(f"  Pass 2: rerank                 {rr.n_passed}/{rr.n_eligible}  "
          f"= {rr.precision_at_k:.1%}")
    if fr.n_eligible:
        print(f"  Pass 3: filtered + rerank      {fr.n_passed}/{fr.n_eligible}  "
              f"= {fr.precision_at_k:.1%}")
    if ar.n_eligible:
        print(f"  Pass 4: agent + rerank         {ar.n_passed}/{ar.n_eligible}  "
              f"= {ar.precision_at_k:.1%}")
    print()
    print(f"  Rerank lift (pass 2 vs 1):                  {summary.rerank_lift_points:+.1f} pp")
    if fr.n_eligible:
        print(f"  Filter lift (pass 3 vs 2 on same queries):  {summary.filter_lift_points:+.1f} pp")
    if ar.n_eligible:
        print(f"  Agent lift (pass 4 vs 2):                   {summary.agent_lift_points:+.1f} pp")
    print()
    print("  By category:")
    for cat, stats in sorted(summary.by_category.items()):
        line = (f"    {cat:20s} no_rerank={stats['passed_no_rerank']}/{stats['n']}  "
                f"rerank={stats['passed_rerank']}/{stats['n']}")
        if stats["n_filterable"]:
            line += (f"  filtered={stats['passed_filtered_rerank']}"
                     f"/{stats['n_filterable']}")
        if stats["n_agent"]:
            line += (f"  agent={stats['passed_agent_rerank']}/{stats['n_agent']}")
        print(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="tests/eval_queries.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-sleep", action="store_true",
                        help="Skip pacing between Cohere calls. Paid key only.")
    parser.add_argument("--skip-agent", action="store_true",
                        help="Skip the agent pass (v3 behaviour).")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON to stdout")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    queries_path = project_root / args.queries
    if not queries_path.exists():
        print(f"ERROR: queries file not found at {queries_path}", file=sys.stderr)
        sys.exit(1)

    with open(queries_path, encoding="utf-8") as f:
        queries_doc = json.load(f)
    queries = queries_doc["queries"]

    pipeline = RetrievalPipeline()
    print(f"Loaded {len(queries)} queries from {queries_path.name}", file=sys.stderr)

    n_filtered = sum(1 for q in queries if q.get("aspect_filter"))
    n_agent = 0 if args.skip_agent else len(queries)
    n_rerank_calls = len(queries) + n_filtered + n_agent
    sleep_seconds = 0.0 if args.no_sleep else RERANK_SLEEP_SECONDS
    estimated_minutes = (n_rerank_calls - 1) * sleep_seconds / 60.0

    total_calls = len(queries) * 2 + n_filtered + n_agent
    print(f"Will run {total_calls} retrieval calls "
          f"({len(queries)} no_rerank + {len(queries)} rerank + "
          f"{n_filtered} filtered_rerank + {n_agent} agent_rerank)",
          file=sys.stderr)
    print(f"Cohere usage: {n_rerank_calls} rerank calls", file=sys.stderr)
    if n_agent:
        print(f"Haiku usage: {n_agent} parse calls (~$0.005 total)",
              file=sys.stderr)
    if sleep_seconds > 0:
        print(f"Pacing: {sleep_seconds}s between rerank calls "
              f"(~{estimated_minutes:.1f} min total wait)", file=sys.stderr)
    print(file=sys.stderr)

    results, summary = run_eval(
        pipeline, queries, top_k=args.top_k,
        sleep_between_rerank=sleep_seconds,
        skip_agent=args.skip_agent,
    )

    if args.json:
        out = {
            "summary": asdict(summary),
            "results": [
                {**asdict(r), "runs": [asdict(run) for run in r.runs]}
                for r in results
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print_human(results, summary)


if __name__ == "__main__":
    main()