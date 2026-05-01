"""
clean_reviews.py — Stage 1 of the Shisha IQ pipeline.

Input:  one or more raw Outscraper CSVs (one per scrape job)
Output: a single canonical reviews.parquet with one row per review.

Operations:
  1. Concatenate all input CSVs
  2. Drop rows with no review_text
  3. Drop non-English reviews (heuristic - the lang filter on Outscraper handles most)
  4. Resolve each row's place_id -> lounge_id using the MVV lounges CSV
  5. Parse review_datetime_utc, compute recency_weight (exp decay, 18mo half-life)
  6. Keep structured Google aspect signals as separate columns
  7. Output canonical schema

Canonical schema (one row per review):
  review_id              str   - Outscraper review_id, primary key
  lounge_id              str   - canonical from MVV CSV
  lounge_name            str   - human-readable
  area                   str   - North/East/West/South/Central
  neighbourhood          str
  price_tier             str   - budget / mid / premium
  price_estimate_gbp     int
  review_text            str
  review_rating          int   - 1-5 stars
  review_date            date
  recency_weight         float - exp(-age_months / 18)
  author_id              str
  author_reviews_count   int
  # Google's structured aspect signals (may be null)
  g_service              float
  g_atmosphere           float
  g_food                 float
  g_wait_time            str
  g_price_per_person     str
  g_noise_level          str
  g_group_size           str
"""

from __future__ import annotations
import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


GOOGLE_STRUCTURED_COLS = {
    "review_questions_Service":          "g_service",
    "review_questions_Atmosphere":       "g_atmosphere",
    "review_questions_Food":             "g_food",
    "review_questions_Wait time":        "g_wait_time",
    "review_questions_Price per person": "g_price_per_person",
    "review_questions_Noise level":      "g_noise_level",
    "review_questions_Group size":       "g_group_size",
}

KEEP_COLS = [
    "review_id", "lounge_id", "lounge_name", "area", "neighbourhood",
    "price_tier", "price_estimate_gbp",
    "review_text", "review_rating", "review_date", "recency_weight",
    "author_id", "author_reviews_count",
    *GOOGLE_STRUCTURED_COLS.values(),
]


def load_lounge_metadata(mvv_csv: Path) -> pd.DataFrame:
    """Load the MVV lounges file - we'll join on place_id once we know it.

    Outscraper returns place_id per row. We don't have place_id pre-baked
    in the MVV CSV, so we resolve via fuzzy name match instead.
    """
    df = pd.read_csv(mvv_csv)
    return df


# Deterministic mapping from Outscraper place_id -> canonical lounge_id.
# We build this from the original Outscraper query (which we control) by
# mapping query -> lounge_id, and then learning query -> place_id from the
# scraped data. Avoids the brittleness of fuzzy name matching.
QUERY_TO_LOUNGE_ID = {
    # Tigerbay's first scrape used a Google Place ID as the query
    "0x48761146d1fab4e3:0xc5b0e79e253ecf85":         "tigerbay_kingsbury",
    "Tigerbay Shisha Lounge Kingsbury":             "tigerbay_kingsbury",
    "Noya Shisha Lounge North London":              "noya_harringay",
    "Laika Shisha Lounge Frith Street Soho":        "laika_soho",
    "Mamounia Lounge Mayfair Curzon Street":        "mamounia_mayfair",
    "Drunch Mayfair Woodstock Street":              "drunch_mayfair",
    "Shishawi Edgware Road London":                 "shishawi_edgware",
    "Al-Dar Edgware Road London":                   "aldar_edgware",
    "Basrah Lounge Edgware Road London":            "basrah_edgware",
    "Globe Lounge Forest Gate London":              "globe_lounge_forest_gate",
    "Ground5 Shisha Lounge London":                 "ground5_south",
    "Cafe Cairo Clapham Landor Road":               "cafe_cairo_clapham",
    "The Shisha Garden Edgware London":             "shisha_garden_edgware",
    "The Banc Shisha Lounge Seven Sisters London":  "the_banc_seven_sisters",
    "Smoke Lab Riverside Vauxhall shisha":          "smoke_lab_vauxhall",
}


def resolve_lounge_id(query: str, lounge_meta: pd.DataFrame) -> str | None:
    """Map an Outscraper query (which we control) to a canonical lounge_id.

    Using the query field is deterministic - no fuzzy matching needed.
    Outscraper preserves the exact query string for every row it returns,
    so this is reliable.
    """
    if not isinstance(query, str):
        return None
    return QUERY_TO_LOUNGE_ID.get(query.strip())


def is_likely_english(text: str) -> bool:
    """Cheap English-ish heuristic. The lang=en filter on Outscraper does the
    real work; this is belt-and-braces for stragglers.

    Rule: at least 60% of chars must be ASCII letters/whitespace/punctuation.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    ascii_ish = sum(1 for c in text if ord(c) < 128)
    return ascii_ish / len(text) >= 0.6


def recency_weight(review_date: datetime, half_life_months: float = 18.0) -> float:
    """exp(-age_months / half_life_months). Newer reviews = higher weight."""
    if pd.isna(review_date):
        return 0.0
    if review_date.tzinfo is None:
        review_date = review_date.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    age_days = (now - review_date).days
    age_months = age_days / 30.0
    return math.exp(-age_months / half_life_months)


def clean_one_csv(csv_path: Path, lounge_meta: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    n_raw = len(raw)

    # 1. Drop rows with no review text
    raw = raw[raw["review_text"].notna() & raw["review_text"].astype(str).str.strip().ne("")]
    n_after_text = len(raw)

    # 2. English filter (heuristic)
    raw = raw[raw["review_text"].apply(is_likely_english)]
    n_after_lang = len(raw)

    # 3. Resolve lounge_id from the original query (deterministic mapping)
    raw["lounge_id"] = raw["query"].apply(lambda q: resolve_lounge_id(q, lounge_meta))
    unresolved = raw[raw["lounge_id"].isna()]
    if len(unresolved):
        unique_unresolved = unresolved["name"].unique().tolist()
        print(f"  ⚠️ {len(unresolved)} rows unresolved across {len(unique_unresolved)} place(s):")
        for n in unique_unresolved[:5]:
            print(f"      - {n[:100]}")
    raw = raw[raw["lounge_id"].notna()]
    n_after_resolve = len(raw)

    # 4. Join lounge metadata
    meta_keep = ["lounge_id", "name", "area", "neighbourhood", "price_tier", "price_estimate_gbp"]
    meta = lounge_meta[meta_keep].rename(columns={"name": "lounge_name"})
    raw = raw.merge(meta, on="lounge_id", how="left")

    # 5. Parse date + recency weight
    raw["review_date"] = pd.to_datetime(raw["review_datetime_utc"], errors="coerce", utc=True)
    raw["recency_weight"] = raw["review_date"].apply(recency_weight)

    # 6. Pull Google structured fields with renamed columns, fill missing as NaN
    for src, dst in GOOGLE_STRUCTURED_COLS.items():
        raw[dst] = raw[src] if src in raw.columns else pd.NA

    # 7. Keep only the columns we care about, in canonical order
    out = raw[KEEP_COLS].copy()

    print(f"  rows: {n_raw} raw -> {n_after_text} with text -> "
          f"{n_after_lang} English -> {n_after_resolve} resolved -> {len(out)} final")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="One or more raw Outscraper CSV paths")
    parser.add_argument("--lounges", required=True,
                        help="Path to shisha_iq_mvv_lounges.csv")
    parser.add_argument("--out", required=True,
                        help="Output CSV path")
    args = parser.parse_args()

    lounge_meta = load_lounge_metadata(Path(args.lounges))
    print(f"Loaded {len(lounge_meta)} lounges from MVV metadata")

    cleaned_frames = []
    for p in args.inputs:
        path = Path(p)
        print(f"\nProcessing {path.name}")
        cleaned_frames.append(clean_one_csv(path, lounge_meta))

    final = pd.concat(cleaned_frames, ignore_index=True)

    # Dedupe on review_id (in case the same scrape was run twice)
    n_before = len(final)
    final = final.drop_duplicates(subset=["review_id"], keep="first")
    if n_before != len(final):
        print(f"\nDropped {n_before - len(final)} duplicate review_ids")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False)

    print(f"\n=== DONE ===")
    print(f"Wrote {len(final)} reviews to {out_path}")
    print(f"\nPer-lounge counts:")
    print(final.groupby("lounge_id").size().sort_values(ascending=False).to_string())
    print(f"\nDate range: {final['review_date'].min()} to {final['review_date'].max()}")
    print(f"Mean recency_weight: {final['recency_weight'].mean():.3f}")
    print(f"Reviews with Google service rating: {final['g_service'].notna().sum()}")


if __name__ == "__main__":
    main()