"""
absa_pilot_v2.py - second iteration of the ABSA pilot.

Changes from v1:
  1. Parser now extracts the first valid JSON OBJECT from the response,
     regardless of code fences or trailing prose. Fixes review #7's
     "JSON then explanation" failure.
  2. Quote validation uses Unicode NFKC normalisation before substring
     check. Fixes false-negative on em-dash / smart-quote variants.
  3. Prompt now explicitly forbids commentary after the JSON, and asks
     for distinct quotes when multiple aspects share evidence.

Re-runs on the same stratified sample so we can compare directly to v1.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------------------------
# Aspect taxonomy - unchanged from v1
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
# Prompt v2 - tightened on commentary and quote distinctness
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
# Robust JSON extraction
# ---------------------------------------------------------------------------
def extract_first_json_object(raw: str) -> dict | None:
    """Find and parse the first balanced JSON object in raw text.

    Handles all of:
      - Pure JSON: '{"aspects": []}'
      - Code-fenced JSON: '```json\\n{"aspects": []}\\n```'
      - JSON followed by commentary: '{"aspects": []}\\n\\nThis review is...'
      - Code-fenced JSON followed by commentary
      - Leading whitespace / preamble before JSON

    Returns the parsed dict, or None if no valid object is found.
    """
    # Walk the string looking for the first '{', then track brace depth
    # respecting string literals and escapes. When depth returns to zero,
    # try to parse the slice. If it parses, return it. If not, keep walking.
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
                        break  # try next '{' in outer loop
        i += 1
    return None


# ---------------------------------------------------------------------------
# Unicode-aware quote validation
# ---------------------------------------------------------------------------
def normalise_for_compare(s: str) -> str:
    """Normalise a string for substring comparison.

    Unifies Unicode forms (NFKC), then maps common punctuation variants
    (smart quotes, em/en dashes, ellipsis) to ASCII equivalents.
    """
    s = unicodedata.normalize("NFKC", s)
    replacements = {
        "\u2018": "'", "\u2019": "'",  # smart single quotes
        "\u201C": '"', "\u201D": '"',  # smart double quotes
        "\u2013": "-", "\u2014": "-",  # en/em dashes
        "\u2026": "...",                # ellipsis
        "\u00A0": " ",                  # non-breaking space
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s


def validate_quotes(parsed: dict, review_text: str) -> list[dict]:
    """Mark each aspect with quote_valid=True if its quote (normalised) is
    a substring of the review text (also normalised)."""
    review_norm = normalise_for_compare(review_text)
    out = []
    for item in parsed.get("aspects", []):
        item_copy = dict(item)
        quote = item_copy.get("quote", "")
        item_copy["quote_valid"] = bool(quote) and normalise_for_compare(quote) in review_norm
        out.append(item_copy)
    return out


def validate_aspects(parsed: dict) -> list[str]:
    return [
        item["aspect"]
        for item in parsed.get("aspects", [])
        if item.get("aspect") not in ASPECT_TAXONOMY
    ]


# ---------------------------------------------------------------------------
# Pilot driver
# ---------------------------------------------------------------------------
def call_haiku(client: OpenAI, system: str, review_text: str) -> tuple[str, dict]:
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


def stratified_sample(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
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
    print(f"Loaded {len(df)} reviews | sampled {len(sample)} for pilot v2\n")

    system_prompt = build_system_prompt()

    total_in, total_out = 0, 0
    n_parse_fail = 0
    n_invalid_aspect = 0
    n_quote_fail = 0
    n_total_aspects = 0
    n_quote_dup = 0
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

        parsed = extract_first_json_object(raw)
        if parsed is None:
            print(f"  JSON PARSE FAIL\n  raw: {raw[:200]}\n")
            n_parse_fail += 1
            continue

        invalid = validate_aspects(parsed)
        validated = validate_quotes(parsed, review_text)

        # Detect any quote duplicated across aspects within this review
        quotes = [a.get("quote", "") for a in validated]
        dups_here = len(quotes) - len(set(quotes))
        n_quote_dup += dups_here

        n_total_aspects += len(validated)
        n_invalid_aspect += len(invalid)
        n_quote_fail += sum(1 for a in validated if not a["quote_valid"])

        # Detect trailing prose - useful diagnostic even when parse succeeds
        match = re.search(r"\}\s*\n", raw)
        trailing = ""
        if match:
            after = raw[match.end():].strip()
            if after and not after.startswith("```"):
                trailing = after[:80]

        print(f"  EXTRACTED ({len(validated)} aspects):")
        for a in validated:
            tick = "\u2713" if a["quote_valid"] else "\u2717"
            in_tax = "" if a["aspect"] in ASPECT_TAXONOMY else " [INVALID ASPECT]"
            print(f"    {tick} {a['aspect']:20s} {a['sentiment']:10s}{in_tax}")
            print(f"      \"{a['quote']}\"")
        if invalid:
            print(f"  WARNING: aspects outside taxonomy: {invalid}")
        if dups_here:
            print(f"  WARNING: {dups_here} duplicate quote(s) across aspects")
        if trailing:
            print(f"  NOTE: trailing prose detected after JSON: \"{trailing}...\"")
        print(f"  tokens: {usage['in']} in / {usage['out']} out\n")

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

        time.sleep(0.5)

    print("=" * 60)
    print("PILOT v2 SUMMARY")
    print("=" * 60)
    print(f"Reviews processed:        {len(results)}/{len(sample)}")
    print(f"JSON parse failures:      {n_parse_fail}")
    print(f"Aspects extracted:        {n_total_aspects}")
    print(f"  out-of-taxonomy:        {n_invalid_aspect}")
    print(f"  quote validation fail:  {n_quote_fail}")
    print(f"  duplicate quotes:       {n_quote_dup}")
    print(f"Tokens: {total_in} in / {total_out} out")

    cost_usd = total_in / 1_000_000 * 1.0 + total_out / 1_000_000 * 5.0
    print(f"Estimated cost:           ${cost_usd:.4f}")
    if results:
        print(f"Projected full-run cost (3990 reviews): "
              f"${cost_usd * 3990 / len(results):.2f}")

    out_path = project_root / "data" / "clean" / "absa_pilot_v2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()