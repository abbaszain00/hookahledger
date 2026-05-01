"""
absa_batch_async.py - run ABSA via Anthropic's Message Batches API.

50% cheaper than sync. Async - submit now, results within 24h (usually
much faster). Sidesteps rate limit problems because Anthropic handles
concurrency on their side.

Three subcommands:
  submit   - bundle remaining unprocessed reviews into a batch and send
             it to Anthropic. Saves the returned batch_id locally.
  status   - check the batch's progress (counts of processing / succeeded
             / errored / etc).
  collect  - when status reports 'ended', download the results, parse
             them with the same JSON extractor as the sync script, and
             append rows to aspects_long.csv. Records processed
             review_ids in the resume log so a future sync run wouldn't
             redo them.

Reuses the validated v2 prompt and helpers (extract_first_json_object,
quote validation, taxonomy filter) so output is identical in shape to
the sync pipeline.

Usage:
  python scripts/absa_batch_async.py submit
  python scripts/absa_batch_async.py status
  python scripts/absa_batch_async.py collect
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv


MODEL = "claude-haiku-4-5"

ASPECT_TAXONOMY = {
    "flavour_quality": "Taste, freshness, brand variety, specific flavours mentioned (mint, blueberry, etc), how flavourful the smoke is.",
    "coal_management": "Heat quality, coal refresh frequency, type of coal used, how well shisha 'lasts', how 'smooth' or 'well-maintained' the shisha is, mentions of foil/smoke duration. Phrases like 'lasted 2 hours', 'smoked well throughout', 'smooth shisha', 'had to ask for coal refill' all count.",
    "service_speed": "Staff attentiveness, friendliness, professionalism, named staff praise/criticism, how attentive the team is.",
    "value_for_money": "Price relative to quality. Specific prices mentioned (e.g. '£25 a head', 'overpriced', 'good value', 'expensive for what you get').",
    "atmosphere_vibe": "Music, lighting, decor, noise level, energy of the room, ambiance.",
    "seating_comfort": "Furniture, space, indoor/outdoor setup, cramped vs spacious, terrace/garden.",
    "food_quality": "Food menu presence and quality, specific dishes mentioned, food preparation standards.",
    "wait_time": "Queue, booking difficulty, time spent waiting for shisha/food/coal/staff attention.",
}

# Batch pricing (50% off standard Haiku 4.5: $1/$5 -> $0.50/$2.50 per M tokens)
PRICE_IN_PER_M = 0.50
PRICE_OUT_PER_M = 2.50

ASPECT_FIELDS = [
    "review_id", "lounge_id", "review_date", "recency_weight",
    "aspect", "sentiment", "quote", "quote_valid",
]

# Anthropic limits batch custom_id to 64 chars and to a restricted alphabet
# (alphanumeric + underscore + hyphen). Outscraper review_ids are too long
# and contain characters like '/' that would break this. So we send
# enumerated IDs in the batch and keep a sidecar map back to the real ID.
DEFAULT_BATCH_DIR = Path("data/clean/batches")


# ---------------------------------------------------------------------------
# System prompt - same as sync v2
# ---------------------------------------------------------------------------
def build_system_prompt() -> str:
    taxonomy_block = "\n".join(
        f"  - {key}: {desc}" for key, desc in ASPECT_TAXONOMY.items()
    )
    return f"""You are an aspect-based sentiment analysis system for London shisha lounge reviews. Your job is to extract structured aspect-sentiment pairs from each review.

<aspect_taxonomy>
You may ONLY use these eight aspect labels. Do not invent new ones.
{taxonomy_block}
</aspect_taxonomy>

<extraction_rules>
1. Extract every aspect from the taxonomy that the review mentions SPECIFICALLY. Generic phrases like "great place", "good vibes", "would recommend" do NOT count - they describe nothing concrete.
2. For each aspect mentioned, pick a sentiment: "positive", "negative", or "mixed". Use "mixed" only when the review praises and criticises the SAME aspect.
3. For each aspect, include a verbatim quote from the review that supports it. The quote must be an EXACT substring of the review text - do not paraphrase, do not summarise, do not translate, do not normalise punctuation.
4. If two aspects share supporting evidence in the review, pick a DIFFERENT verbatim snippet for each aspect where possible. Do not reuse the same quote across aspects.
5. Cap output at 5 aspects per review. If more are mentioned, pick the 5 most strongly expressed.
6. If the review mentions zero aspects from the taxonomy (e.g. "Was good", or it's about a different business entirely), return an empty list.
7. Do NOT infer aspects that aren't explicitly discussed. If a review only mentions food, do not assume the shisha was good.
</extraction_rules>

<output_format>
Reply with valid JSON ONLY. No markdown fences, no preamble, no commentary, no explanation before or after the JSON. Your entire response must be parseable as a single JSON object.

Schema:
{{
  "aspects": [
    {{"aspect": "<one of the taxonomy keys>", "sentiment": "positive|negative|mixed", "quote": "<verbatim substring>"}}
  ]
}}

If no aspects apply, return exactly: {{"aspects": []}}
</output_format>

<examples>
<example>
<review>Mint flavour was incredible but we waited 40 minutes for coal. Staff were friendly though. £25 a head felt fair.</review>
<output>{{"aspects": [{{"aspect": "flavour_quality", "sentiment": "positive", "quote": "Mint flavour was incredible"}}, {{"aspect": "wait_time", "sentiment": "negative", "quote": "we waited 40 minutes for coal"}}, {{"aspect": "service_speed", "sentiment": "positive", "quote": "Staff were friendly"}}, {{"aspect": "value_for_money", "sentiment": "positive", "quote": "\u00a325 a head felt fair"}}]}}</output>
</example>

<example>
<review>Was good</review>
<output>{{"aspects": []}}</output>
</example>

<example>
<review>The shisha lasted nearly 2 hours and the heat stayed strong throughout. Coals were swapped twice without us asking. Best smoke I've had in a while.</review>
<output>{{"aspects": [{{"aspect": "coal_management", "sentiment": "positive", "quote": "shisha lasted nearly 2 hours and the heat stayed strong throughout"}}, {{"aspect": "service_speed", "sentiment": "positive", "quote": "Coals were swapped twice without us asking"}}, {{"aspect": "flavour_quality", "sentiment": "positive", "quote": "Best smoke I've had in a while"}}]}}</output>
</example>

<example>
<review>Tiny seating area, crushed between people. Shisha is INSANELY OVERPRICED. Don't waste your money.</review>
<output>{{"aspects": [{{"aspect": "seating_comfort", "sentiment": "negative", "quote": "Tiny seating area, crushed between people"}}, {{"aspect": "value_for_money", "sentiment": "negative", "quote": "Shisha is INSANELY OVERPRICED"}}]}}</output>
</example>
</examples>"""


# ---------------------------------------------------------------------------
# Helpers reused from sync pipeline
# ---------------------------------------------------------------------------
def extract_first_json_object(raw: str) -> dict | None:
    n = len(raw)
    i = 0
    while i < n:
        if raw[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        for j in range(i, n):
            ch = raw[j]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[i:j + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        i += 1
    return None


def normalise_for_compare(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-",
        "\u2026": "...",
        "\u00A0": " ",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s


def quote_is_valid(quote: str, review_text: str) -> bool:
    return bool(quote) and normalise_for_compare(quote) in normalise_for_compare(review_text)


# ---------------------------------------------------------------------------
# Loading processed IDs (same source of truth as sync script)
# ---------------------------------------------------------------------------
def load_processed_ids(aspects_csv: Path, processed_log: Path) -> set[str]:
    done: set[str] = set()
    if processed_log.exists():
        with open(processed_log, encoding="utf-8") as f:
            done.update(line.strip() for line in f if line.strip())
    if aspects_csv.exists():
        try:
            existing = pd.read_csv(aspects_csv, usecols=["review_id"])
            done.update(existing["review_id"].astype(str).tolist())
        except Exception:
            pass
    return done


# ---------------------------------------------------------------------------
# Subcommand: submit
# ---------------------------------------------------------------------------
def cmd_submit(args: argparse.Namespace, client: anthropic.Anthropic, project_root: Path) -> None:
    reviews_path = project_root / args.reviews
    aspects_path = project_root / args.aspects
    log_path = project_root / args.processed_log
    batch_dir = project_root / args.batch_dir
    batch_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(reviews_path)
    print(f"Loaded {len(df)} reviews from {reviews_path}")

    done_ids = load_processed_ids(aspects_path, log_path)
    todo = df[~df["review_id"].astype(str).isin(done_ids)].copy()
    print(f"Already processed: {len(done_ids)}, todo: {len(todo)}")
    if args.limit is not None:
        todo = todo.head(args.limit)
        print(f"--limit {args.limit} applied -> submitting {len(todo)}")

    if len(todo) == 0:
        print("Nothing to submit.")
        return
    if len(todo) > 100_000:
        print(f"ERROR: batch size {len(todo)} exceeds Anthropic limit (100k)")
        sys.exit(1)

    system_prompt = build_system_prompt()

    # Build the request list. Use enumerated short IDs (b0, b1, ...) for
    # custom_id and keep a sidecar map back to the real review_id.
    requests = []
    id_map: dict[str, str] = {}
    for i, row in enumerate(todo.itertuples(index=False)):
        custom_id = f"b{i}"
        id_map[custom_id] = row.review_id
        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": MODEL,
                "max_tokens": 800,
                "temperature": 0,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": f"<review>{row.review_text}</review>"},
                ],
            },
        })

    print(f"\nSubmitting batch of {len(requests)} requests to Anthropic...")
    batch = client.messages.batches.create(requests=requests)
    print(f"  batch_id: {batch.id}")
    print(f"  status:   {batch.processing_status}")

    # Persist batch id, id map, and todo manifest. Without these we can't
    # link results back to review rows on collect.
    state = {
        "batch_id": batch.id,
        "submitted_at": batch.created_at.isoformat() if batch.created_at else None,
        "n_requests": len(requests),
        "id_map": id_map,
    }
    state_path = batch_dir / f"{batch.id}.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    # Also save the review rows we submitted - we need recency_weight,
    # lounge_id, etc. on collect. CSV with index by custom_id.
    submitted = todo.copy()
    submitted["custom_id"] = [f"b{i}" for i in range(len(submitted))]
    submitted_path = batch_dir / f"{batch.id}_reviews.csv"
    submitted.to_csv(submitted_path, index=False)

    # Pointer file - so 'status' / 'collect' know which batch to use by
    # default.
    pointer_path = batch_dir / "latest_batch.txt"
    pointer_path.write_text(batch.id, encoding="utf-8")

    print(f"\nState saved to: {state_path}")
    print(f"Reviews snapshot: {submitted_path}")
    print(f"Pointer: {pointer_path}")
    print(f"\nNext: python scripts/absa_batch_async.py status")


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------
def cmd_status(args: argparse.Namespace, client: anthropic.Anthropic, project_root: Path) -> None:
    batch_id = args.batch_id or _read_latest_pointer(project_root, args.batch_dir)
    if not batch_id:
        print("No batch_id given and no latest_batch.txt found.")
        sys.exit(1)

    batch = client.messages.batches.retrieve(batch_id)
    print(f"Batch:      {batch.id}")
    print(f"Status:     {batch.processing_status}")
    print(f"Created:    {batch.created_at}")
    if batch.ended_at:
        print(f"Ended:      {batch.ended_at}")
    if batch.expires_at:
        print(f"Expires:    {batch.expires_at}")
    counts = batch.request_counts
    if counts:
        print(f"\nRequest counts:")
        print(f"  processing: {counts.processing}")
        print(f"  succeeded:  {counts.succeeded}")
        print(f"  errored:    {counts.errored}")
        print(f"  canceled:   {counts.canceled}")
        print(f"  expired:    {counts.expired}")
    if batch.processing_status == "ended":
        print("\nBatch is done. Run: python scripts/absa_batch_async.py collect")
    else:
        print("\nStill running. Re-run this command in a few minutes.")


# ---------------------------------------------------------------------------
# Subcommand: collect
# ---------------------------------------------------------------------------
def cmd_collect(args: argparse.Namespace, client: anthropic.Anthropic, project_root: Path) -> None:
    batch_id = args.batch_id or _read_latest_pointer(project_root, args.batch_dir)
    if not batch_id:
        print("No batch_id given and no latest_batch.txt found.")
        sys.exit(1)

    batch_dir = project_root / args.batch_dir
    state_path = batch_dir / f"{batch_id}.json"
    submitted_path = batch_dir / f"{batch_id}_reviews.csv"
    if not state_path.exists():
        print(f"State file missing: {state_path}")
        sys.exit(1)
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    submitted = pd.read_csv(submitted_path).set_index("custom_id")

    # Confirm it's actually done before we try to fetch results
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"Batch not ended yet (status={batch.processing_status}). "
              "Run 'status' until it reports 'ended'.")
        sys.exit(1)

    aspects_path = project_root / args.aspects
    failures_path = project_root / args.failures
    log_path = project_root / args.processed_log
    aspects_path.parent.mkdir(parents=True, exist_ok=True)

    new_aspects_file = not aspects_path.exists()
    new_failures_file = not failures_path.exists()
    aspects_f = open(aspects_path, "a", newline="", encoding="utf-8")
    failures_f = open(failures_path, "a", newline="", encoding="utf-8")
    log_f = open(log_path, "a", encoding="utf-8")
    aspects_writer = csv.DictWriter(aspects_f, fieldnames=ASPECT_FIELDS)
    failures_writer = csv.DictWriter(
        failures_f, fieldnames=["review_id", "reason", "raw_response"]
    )
    if new_aspects_file:
        aspects_writer.writeheader()
    if new_failures_file:
        failures_writer.writeheader()

    counters = {"ok": 0, "fail": 0, "n_aspects": 0, "tokens_in": 0, "tokens_out": 0}
    n_seen = 0
    try:
        for entry in client.messages.batches.results(batch_id):
            n_seen += 1
            custom_id = entry.custom_id
            if custom_id not in submitted.index:
                # This shouldn't happen unless state files are out of sync
                print(f"  WARNING: result for unknown custom_id {custom_id}")
                continue
            row = submitted.loc[custom_id]
            review_id = str(row["review_id"])
            review_text = row["review_text"]

            if entry.result.type != "succeeded":
                err = getattr(entry.result, "error", None)
                reason = f"{entry.result.type}: {err}"
                counters["fail"] += 1
                failures_writer.writerow({
                    "review_id": review_id, "reason": reason, "raw_response": "",
                })
                continue

            msg = entry.result.message
            counters["tokens_in"] += msg.usage.input_tokens if msg.usage else 0
            counters["tokens_out"] += msg.usage.output_tokens if msg.usage else 0

            # Concatenate text content blocks (usually just one for our prompt)
            raw = "".join(b.text for b in msg.content if b.type == "text")
            parsed = extract_first_json_object(raw)
            if parsed is None or "aspects" not in parsed:
                counters["fail"] += 1
                failures_writer.writerow({
                    "review_id": review_id, "reason": "json_parse_fail",
                    "raw_response": raw[:500],
                })
                continue

            counters["ok"] += 1
            for item in parsed.get("aspects", []):
                aspect = item.get("aspect", "")
                sentiment = item.get("sentiment", "")
                quote = item.get("quote", "")
                if aspect not in ASPECT_TAXONOMY:
                    continue
                if sentiment not in ("positive", "negative", "mixed"):
                    continue
                aspects_writer.writerow({
                    "review_id": review_id,
                    "lounge_id": row["lounge_id"],
                    "review_date": row["review_date"],
                    "recency_weight": row["recency_weight"],
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "quote": quote,
                    "quote_valid": quote_is_valid(quote, review_text),
                })
                counters["n_aspects"] += 1
            log_f.write(review_id + "\n")

            if n_seen % 250 == 0:
                aspects_f.flush()
                failures_f.flush()
                log_f.flush()
                print(f"  ... {n_seen} results processed")
    finally:
        aspects_f.close()
        failures_f.close()
        log_f.close()

    cost = (counters["tokens_in"] / 1_000_000 * PRICE_IN_PER_M
            + counters["tokens_out"] / 1_000_000 * PRICE_OUT_PER_M)
    print("\n" + "=" * 60)
    print("BATCH COLLECT SUMMARY")
    print("=" * 60)
    print(f"Results streamed:    {n_seen}")
    print(f"Reviews ok / failed: {counters['ok']} / {counters['fail']}")
    print(f"Aspects extracted:   {counters['n_aspects']}")
    if counters["ok"]:
        print(f"Avg aspects/review:  {counters['n_aspects'] / counters['ok']:.2f}")
    print(f"Tokens:              {counters['tokens_in']} in / {counters['tokens_out']} out")
    print(f"Cost (batch rate):   ${cost:.3f}")
    print(f"\nAspect rows: {aspects_path}")
    if counters["fail"]:
        print(f"Failures:    {failures_path}")


# ---------------------------------------------------------------------------
# Pointer helper
# ---------------------------------------------------------------------------
def _read_latest_pointer(project_root: Path, batch_dir_arg: str) -> str | None:
    pointer = project_root / batch_dir_arg / "latest_batch.txt"
    if not pointer.exists():
        return None
    return pointer.read_text(encoding="utf-8").strip() or None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    common_paths = [
        ("--reviews", "data/clean/reviews_clean.csv", "Path to cleaned reviews CSV"),
        ("--aspects", "data/clean/aspects_long.csv", "Output path for aspect rows"),
        ("--failures", "data/clean/absa_failures.csv", "Output path for failed reviews"),
        ("--processed-log", "data/clean/absa_processed.txt", "Resume log of processed review_ids"),
        ("--batch-dir", "data/clean/batches", "Directory for batch state files"),
    ]

    p_submit = sub.add_parser("submit", help="Submit a new batch")
    for flag, default, help_ in common_paths:
        p_submit.add_argument(flag, default=default, help=help_)
    p_submit.add_argument("--limit", type=int, default=None,
                          help="Cap number of reviews submitted (for testing)")

    p_status = sub.add_parser("status", help="Check batch status")
    p_status.add_argument("--batch-id", default=None,
                          help="Override the latest batch pointer")
    p_status.add_argument("--batch-dir", default="data/clean/batches")

    p_collect = sub.add_parser("collect", help="Download and parse batch results")
    for flag, default, help_ in common_paths:
        p_collect.add_argument(flag, default=default, help=help_)
    p_collect.add_argument("--batch-id", default=None,
                           help="Override the latest batch pointer")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not in .env")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    if args.cmd == "submit":
        cmd_submit(args, client, project_root)
    elif args.cmd == "status":
        cmd_status(args, client, project_root)
    elif args.cmd == "collect":
        cmd_collect(args, client, project_root)


if __name__ == "__main__":
    main()