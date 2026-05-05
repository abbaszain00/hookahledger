"""
absa_pilot.py - test the ABSA prompt on a stratified sample of real reviews.

Goals of the pilot:
  1. See how Haiku handles real review variety (terse, verbose, off-topic, slang).
  2. Validate that returned JSON parses cleanly.
  3. Validate that quotes are real substrings of the review.
  4. Confirm aspects stay within the taxonomy.
  5. Confirm multiple aspects per review get extracted (not just one).

This is NOT the production batch script - it's a 10-review eyeball test
so we can iterate the prompt before spending money on the full 3,990-review run.

Prompt design is grounded in:
  - Anthropic prompt engineering best practices: XML tags, clear role,
    structured examples, output format spec, few-shot.
  - Academic ABSA literature: cap aspects per review, explicit taxonomy,
    salience filter, JSON output, temperature 0.
  - Real review variety from this project's own data.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------------------------
# Aspect taxonomy - matches the Day 1 design doc
# ---------------------------------------------------------------------------
ASPECT_TAXONOMY = {
    "flavour_quality": "Taste, freshness, brand variety, specific flavours mentioned (mint, blueberry, etc), how flavourful the smoke is.",
    "coal_management": "Heat quality, coal refresh frequency, type of coal used, how well shisha 'lasts', mentions of foil/smoke duration. Phrases like 'lasted 2 hours', 'smoked well throughout', 'had to ask for coal refill' all count.",
    "service_speed": "Staff attentiveness, friendliness, professionalism, named staff praise/criticism, how attentive the team is.",
    "value_for_money": "Price relative to quality. Specific prices mentioned (e.g. '£25 a head', 'overpriced', 'good value', 'expensive for what you get').",
    "atmosphere_vibe": "Music, lighting, decor, noise level, energy of the room, ambiance.",
    "seating_comfort": "Furniture, space, indoor/outdoor setup, cramped vs spacious, terrace/garden.",
    "food_quality": "Food menu presence and quality, specific dishes mentioned, food preparation standards.",
    "wait_time": "Queue, booking difficulty, time spent waiting for shisha/food/coal/staff attention.",
}


# ---------------------------------------------------------------------------
# The prompt - SYSTEM message
# Design principles applied:
#   - Clear role
#   - XML tags to separate sections (Anthropic best practice)
#   - Explicit aspect taxonomy with vocabulary anchors
#   - Few-shot examples covering edge cases (multiple aspects, off-topic, terse)
#   - Strict output format with no preamble (per AWS Bedrock guidance)
#   - Salience filter (cap at top-5 aspects, only mention specific things)
#   - Verbatim quote requirement for downstream substring validation
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
3. For each aspect, include a verbatim quote from the review that supports it. The quote must be an EXACT substring of the review text - do not paraphrase, do not summarise, do not translate.
4. Cap output at 5 aspects per review. If more are mentioned, pick the 5 most strongly expressed.
5. If the review mentions zero aspects from the taxonomy (e.g. "Was good", or it's about a different business entirely), return an empty list.
6. Do NOT infer aspects that aren't explicitly discussed. If a review only mentions food, do not assume the shisha was good.
</extraction_rules>

<output_format>
Reply with valid JSON only, no preamble, no markdown fences, no explanation. Schema:
{{
  "aspects": [
    {{"aspect": "<one of the taxonomy keys>", "sentiment": "positive|negative|mixed", "quote": "<verbatim substring>"}}
  ]
}}

If no aspects apply, return: {{"aspects": []}}
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
# Pilot run logic
# ---------------------------------------------------------------------------
def call_haiku(client: OpenAI, system: str, review_text: str) -> tuple[str, dict]:
    """Make one ABSA call. Returns (raw_response, usage_dict)."""
    resp = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"<review>{review_text}</review>"},
        ],
        temperature=0,
        max_tokens=800,
    )
    raw = resp.choices[0].message.content
    usage = {
        "in": resp.usage.prompt_tokens if resp.usage else 0,
        "out": resp.usage.completion_tokens if resp.usage else 0,
    }
    return raw, usage


def parse_json(raw: str) -> dict | None:
    """Parse JSON from response, tolerating optional code fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def validate_quotes(parsed: dict, review_text: str) -> list[dict]:
    """For each aspect, check whether the quote is a real substring.

    Returns the aspects list with an added 'quote_valid' bool per item.
    """
    out = []
    for item in parsed.get("aspects", []):
        item_copy = dict(item)
        quote = item_copy.get("quote", "")
        item_copy["quote_valid"] = bool(quote) and quote in review_text
        out.append(item_copy)
    return out


def validate_aspects(parsed: dict) -> list[str]:
    """Return list of aspects in the response that aren't in the taxonomy."""
    return [
        item["aspect"]
        for item in parsed.get("aspects", [])
        if item.get("aspect") not in ASPECT_TAXONOMY
    ]


def stratified_sample(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Pick 10 reviews covering high/mid/long-tail lounges, mixed ratings, mixed lengths."""

    def pick(lounge: str, n: int, criteria: str | None = None) -> pd.DataFrame:
        sub = df[df["lounge_id"] == lounge]
        if criteria == "short":
            sub = sub[sub["review_text"].str.len() < 100]
        elif criteria == "long":
            sub = sub[sub["review_text"].str.len() > 300]
        elif criteria == "negative":
            sub = sub[sub["review_rating"] <= 2]
        return sub.sample(n=min(n, len(sub)), random_state=seed)

    parts = [
        pick("tigerbay_kingsbury", 1, "long"),
        pick("tigerbay_kingsbury", 1, "negative"),
        pick("tigerbay_kingsbury", 1, "short"),
        pick("laika_soho", 1, "long"),
        pick("laika_soho", 1, "negative"),
        pick("mamounia_mayfair", 1, "long"),
        pick("cafe_cairo_clapham", 1),
        pick("noya_harringay", 1, "short"),
        pick("basrah_edgware", 1, "negative"),
        pick("shisha_garden_edgware", 1, "long"),
    ]
    return pd.concat(parts).reset_index(drop=True)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not in .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    df = pd.read_csv(project_root / "data" / "clean" / "reviews_clean.csv")
    sample = stratified_sample(df)
    print(f"Loaded {len(df)} reviews | sampled {len(sample)} for pilot\n")

    system_prompt = build_system_prompt()

    total_in, total_out = 0, 0
    n_parse_fail = 0
    n_invalid_aspect = 0
    n_quote_fail = 0
    n_total_aspects = 0

    results = []
    for i, row in sample.iterrows():
        review_text = row["review_text"]
        lounge = row["lounge_id"]
        rating = row["review_rating"]

        print(f"--- #{i+1} | {lounge} | {rating}\u2605 | {len(review_text)} chars ---")
        print(f"REVIEW: {review_text[:200]}{'...' if len(review_text) > 200 else ''}")

        try:
            raw, usage = call_haiku(client, system_prompt, review_text)
        except Exception as e:
            print(f"  API ERROR: {e}\n")
            continue

        total_in += usage["in"]
        total_out += usage["out"]

        parsed = parse_json(raw)
        if parsed is None:
            print(f"  JSON PARSE FAIL\n  raw: {raw[:200]}\n")
            n_parse_fail += 1
            continue

        invalid = validate_aspects(parsed)
        validated = validate_quotes(parsed, review_text)

        n_total_aspects += len(validated)
        n_invalid_aspect += len(invalid)
        n_quote_fail += sum(1 for a in validated if not a["quote_valid"])

        print(f"  EXTRACTED ({len(validated)} aspects):")
        for a in validated:
            tick = "\u2713" if a["quote_valid"] else "\u2717"
            in_tax = "" if a["aspect"] in ASPECT_TAXONOMY else " [INVALID ASPECT]"
            print(f"    {tick} {a['aspect']:20s} {a['sentiment']:10s}{in_tax}")
            print(f"      \"{a['quote']}\"")
        if invalid:
            print(f"  WARNING: aspects outside taxonomy: {invalid}")
        print(f"  tokens: {usage['in']} in / {usage['out']} out")
        print()

        results.append({
            "review_id": row["review_id"],
            "lounge_id": lounge,
            "review_text": review_text,
            "raw_response": raw,
            "parsed": parsed,
            "validated": validated,
            "invalid_aspects": invalid,
            "tokens_in": usage["in"],
            "tokens_out": usage["out"],
        })

        time.sleep(0.5)  # gentle on rate limits

    # ---------------- Summary ----------------
    print("=" * 60)
    print("PILOT SUMMARY")
    print("=" * 60)
    print(f"Reviews processed:      {len(results)}/{len(sample)}")
    print(f"JSON parse failures:    {n_parse_fail}")
    print(f"Aspects extracted:      {n_total_aspects}")
    print(f"  out-of-taxonomy:      {n_invalid_aspect}")
    print(f"  quote validation fail:{n_quote_fail}")
    print(f"Tokens: {total_in} in / {total_out} out")

    # cost estimate using Haiku 4.5 published pricing ($1/M in, $5/M out)
    cost_usd = total_in / 1_000_000 * 1.0 + total_out / 1_000_000 * 5.0
    print(f"Estimated cost:         ${cost_usd:.4f}")
    print(f"Projected full run cost (3990 reviews): "
          f"${cost_usd * 3990 / len(results):.2f}")

    # Persist for inspection
    out_path = project_root / "data" / "clean" / "absa_pilot_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()