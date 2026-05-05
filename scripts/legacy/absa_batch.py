"""
absa_batch.py - run the validated ABSA prompt across the full review corpus.

Production-ready features:
  - Checkpointing: results are written incrementally to a CSV, indexed by
    review_id. Re-running the script picks up where the last run left off.
  - Concurrency: up to N parallel API calls via ThreadPoolExecutor.
  - Retry: exponential backoff on transient errors (429, 5xx, network).
  - Progress: tqdm bar with running cost / token counters.
  - Structured output: writes ONE ROW PER (review, aspect) pair to
    aspects_long.csv, ready for SQLite ingestion. Also writes a parallel
    failures.csv for any reviews that didn't parse cleanly.

Output schema (aspects_long.csv):
  review_id, lounge_id, review_date, recency_weight,
  aspect, sentiment, quote, quote_valid

Reviews with zero aspects extracted produce zero output rows. They are
recorded in a separate processed_review_ids set so we don't reprocess them.

Usage:
  python scripts/absa_batch.py                 # run on full corpus
  python scripts/absa_batch.py --limit 100     # test on first 100 only
  python scripts/absa_batch.py --workers 8     # tune concurrency
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
from dotenv import load_dotenv
import anthropic
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Aspect taxonomy - same as the validated pilot
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Pricing for cost projection (Haiku 4.5 published rates as of Apr 2026)
# ---------------------------------------------------------------------------
PRICE_IN_PER_M = 1.0   # USD per million input tokens
PRICE_OUT_PER_M = 5.0  # USD per million output tokens

# Output schema for aspects_long.csv
ASPECT_FIELDS = [
    "review_id", "lounge_id", "review_date", "recency_weight",
    "aspect", "sentiment", "quote", "quote_valid",
]


# ---------------------------------------------------------------------------
# System prompt - validated in pilot v2
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
# Helpers (lifted from validated pilot v2)
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
# API call with retry
# ---------------------------------------------------------------------------
def call_with_retry(client: anthropic.Anthropic, system: str, review_text: str,
                    max_retries: int = 4) -> tuple[str, dict]:
    """One ABSA call with exponential backoff on transient errors."""
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5",
                system=system,
                messages=[
                    {"role": "user", "content": f"<review>{review_text}</review>"},
                ],
                temperature=0,
                max_tokens=800,
            )
            raw = resp.content[0].text
            usage = {
                "in": resp.usage.input_tokens,
                "out": resp.usage.output_tokens,
            }
            return raw, usage
        except Exception as e:
            last_err = e
            # Backoff: 2, 4, 8, 16 seconds + jitter
            sleep_for = (2 ** (attempt + 1)) + random.uniform(0, 1)
            time.sleep(sleep_for)
    # All retries exhausted - raise the last error
    raise last_err if last_err else RuntimeError("call_with_retry: no error captured")


# ---------------------------------------------------------------------------
# Per-review processing
# ---------------------------------------------------------------------------
def process_one(client: OpenAI, system: str, row: dict) -> dict:
    """Process one review row. Returns a dict with success/failure info and rows
    to write."""
    review_id = row["review_id"]
    lounge_id = row["lounge_id"]
    review_text = row["review_text"]
    review_date = row["review_date"]
    recency_weight = row["recency_weight"]

    try:
        raw, usage = call_with_retry(client, system, review_text)
    except Exception as e:
        return {
            "review_id": review_id, "ok": False, "reason": f"api_error: {e}",
            "rows": [], "tokens_in": 0, "tokens_out": 0, "raw": "",
        }

    parsed = extract_first_json_object(raw)
    if parsed is None or "aspects" not in parsed:
        return {
            "review_id": review_id, "ok": False, "reason": "json_parse_fail",
            "rows": [], "tokens_in": usage["in"], "tokens_out": usage["out"],
            "raw": raw,
        }

    aspect_rows = []
    for item in parsed.get("aspects", []):
        aspect = item.get("aspect", "")
        sentiment = item.get("sentiment", "")
        quote = item.get("quote", "")
        if aspect not in ASPECT_TAXONOMY:
            # Drop out-of-taxonomy aspects silently rather than failing the whole row
            continue
        if sentiment not in ("positive", "negative", "mixed"):
            continue
        aspect_rows.append({
            "review_id": review_id,
            "lounge_id": lounge_id,
            "review_date": review_date,
            "recency_weight": recency_weight,
            "aspect": aspect,
            "sentiment": sentiment,
            "quote": quote,
            "quote_valid": quote_is_valid(quote, review_text),
        })

    return {
        "review_id": review_id, "ok": True, "reason": "",
        "rows": aspect_rows, "tokens_in": usage["in"], "tokens_out": usage["out"],
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_processed_ids(aspects_csv: Path, processed_log: Path) -> set[str]:
    """Reviews are 'done' if they appear in the processed_log (one id per line)
    OR if they have rows in aspects_csv."""
    done: set[str] = set()
    if processed_log.exists():
        with open(processed_log, encoding="utf-8") as f:
            done.update(line.strip() for line in f if line.strip())
    if aspects_csv.exists():
        try:
            existing = pd.read_csv(aspects_csv, usecols=["review_id"])
            done.update(existing["review_id"].astype(str).tolist())
        except Exception:
            pass  # malformed - just skip
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, only process the first N unprocessed reviews")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of concurrent API calls (default 5)")
    parser.add_argument("--reviews", type=str, default="data/clean/reviews_clean.csv",
                        help="Path to cleaned reviews CSV")
    parser.add_argument("--out", type=str, default="data/clean/aspects_long.csv",
                        help="Output path for aspect rows")
    parser.add_argument("--failures", type=str, default="data/clean/absa_failures.csv",
                        help="Output path for failed reviews")
    parser.add_argument("--processed-log", type=str,
                        default="data/clean/absa_processed.txt",
                        help="Log of processed review_ids (for resume)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not in .env")
        sys.exit(1)

    reviews_path = project_root / args.reviews
    out_path = project_root / args.out
    failures_path = project_root / args.failures
    log_path = project_root / args.processed_log
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(reviews_path)
    print(f"Loaded {len(df)} reviews from {reviews_path}")

    # Resume support: skip anything already processed
    done_ids = load_processed_ids(out_path, log_path)
    if done_ids:
        print(f"Resuming - {len(done_ids)} reviews already processed, skipping them")
    todo_df = df[~df["review_id"].astype(str).isin(done_ids)].copy()
    if args.limit is not None:
        todo_df = todo_df.head(args.limit)
    print(f"Processing {len(todo_df)} reviews this run\n")

    if len(todo_df) == 0:
        print("Nothing to do.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = build_system_prompt()

    # Open output files in append mode. Write headers if files are new.
    new_aspects_file = not out_path.exists()
    new_failures_file = not failures_path.exists()
    aspects_f = open(out_path, "a", newline="", encoding="utf-8")
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

    write_lock = Lock()
    counters = {"ok": 0, "fail": 0, "n_aspects": 0, "tokens_in": 0, "tokens_out": 0}

    rows_iter = todo_df.to_dict("records")
    bar = tqdm(total=len(rows_iter), desc="ABSA", unit="rev")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_one, client, system_prompt, row): row["review_id"]
                for row in rows_iter
            }
            for fut in as_completed(futures):
                result = fut.result()
                with write_lock:
                    counters["tokens_in"] += result["tokens_in"]
                    counters["tokens_out"] += result["tokens_out"]
                    if result["ok"]:
                        counters["ok"] += 1
                        for r in result["rows"]:
                            aspects_writer.writerow(r)
                            counters["n_aspects"] += 1
                        log_f.write(result["review_id"] + "\n")
                    else:
                        counters["fail"] += 1
                        failures_writer.writerow({
                            "review_id": result["review_id"],
                            "reason": result["reason"],
                            "raw_response": (result["raw"] or "")[:500],
                        })
                    # Flush so a crash doesn't lose buffered data
                    aspects_f.flush()
                    failures_f.flush()
                    log_f.flush()

                cost = (counters["tokens_in"] / 1_000_000 * PRICE_IN_PER_M
                        + counters["tokens_out"] / 1_000_000 * PRICE_OUT_PER_M)
                bar.set_postfix({
                    "ok": counters["ok"],
                    "fail": counters["fail"],
                    "aspects": counters["n_aspects"],
                    "cost_usd": f"{cost:.3f}",
                })
                bar.update(1)
    finally:
        bar.close()
        aspects_f.close()
        failures_f.close()
        log_f.close()

    cost = (counters["tokens_in"] / 1_000_000 * PRICE_IN_PER_M
            + counters["tokens_out"] / 1_000_000 * PRICE_OUT_PER_M)

    print("\n" + "=" * 60)
    print("ABSA BATCH SUMMARY")
    print("=" * 60)
    print(f"Reviews processed:   {counters['ok']} ok / {counters['fail']} failed")
    print(f"Aspects extracted:   {counters['n_aspects']}")
    if counters["ok"]:
        print(f"Avg aspects/review:  {counters['n_aspects'] / counters['ok']:.2f}")
    print(f"Tokens:              {counters['tokens_in']} in / {counters['tokens_out']} out")
    print(f"Cost (USD):          ${cost:.3f}")
    print(f"\nAspect rows written: {out_path}")
    if counters["fail"]:
        print(f"Failures logged:     {failures_path}  - rerun the script to retry")
    print(f"Resume log:          {log_path}")


if __name__ == "__main__":
    main()