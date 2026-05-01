"""
build_counts.py - build the deterministic SQLite count store from aspects_long.csv.

This is the FIX for citation hallucination flagged in the Day 1 design doc.
The LLM never invents numbers - the live query layer fetches them from here.

Tables:
  aspect_counts:
    Per (lounge_id, aspect, sentiment) - how many reviews mention this aspect
    with this sentiment. The primary lookup table for evidence cards.
    Note: a single review may contribute multiple rows (different aspects).

  lounge_totals:
    Per lounge_id - total review count, total aspect-mentions, mean recency.
    Useful for normalisation ("47 reviews mention coal at Tigerbay" needs the
    denominator to be meaningful).

  aspect_quotes:
    Top quote per (lounge_id, aspect, sentiment) ranked by recency_weight.
    The single best quote to surface in evidence cards alongside the count.
    Picking deterministically here means the LLM doesn't have to choose.

Counts are deduplicated at the (review_id, aspect, sentiment) level - if a
single review extracts the same aspect+sentiment twice (which happens
occasionally because Haiku sometimes splits a long review into two snippets
on the same aspect), it counts as ONE for the count table. Both snippets
are still preserved in aspect_quotes.

Usage:
  python scripts/build_counts.py
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


SCHEMA = """
DROP TABLE IF EXISTS aspect_counts;
DROP TABLE IF EXISTS lounge_totals;
DROP TABLE IF EXISTS aspect_quotes;

CREATE TABLE aspect_counts (
    lounge_id   TEXT NOT NULL,
    aspect      TEXT NOT NULL,
    sentiment   TEXT NOT NULL,
    n_reviews   INTEGER NOT NULL,
    PRIMARY KEY (lounge_id, aspect, sentiment)
);

CREATE TABLE lounge_totals (
    lounge_id            TEXT PRIMARY KEY,
    total_reviews        INTEGER NOT NULL,
    total_aspect_mentions INTEGER NOT NULL,
    mean_recency_weight  REAL NOT NULL
);

CREATE TABLE aspect_quotes (
    lounge_id      TEXT NOT NULL,
    aspect         TEXT NOT NULL,
    sentiment      TEXT NOT NULL,
    quote          TEXT NOT NULL,
    review_id      TEXT NOT NULL,
    review_date    TEXT,
    recency_weight REAL,
    rank_in_group  INTEGER NOT NULL,
    PRIMARY KEY (lounge_id, aspect, sentiment, rank_in_group)
);

CREATE INDEX idx_counts_lookup ON aspect_counts (lounge_id, aspect);
CREATE INDEX idx_quotes_lookup ON aspect_quotes (lounge_id, aspect, sentiment);
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspects", type=str,
                        default="data/clean/aspects_long.csv",
                        help="Path to aspects_long.csv from absa_batch")
    parser.add_argument("--reviews", type=str,
                        default="data/clean/reviews_clean.csv",
                        help="Path to cleaned reviews CSV (for review counts)")
    parser.add_argument("--db", type=str,
                        default="data/clean/hookahledger.sqlite",
                        help="Output SQLite path")
    parser.add_argument("--top-quotes", type=int, default=3,
                        help="How many top quotes to keep per (lounge, aspect, sentiment)")
    parser.add_argument("--require-valid-quote", action="store_true",
                        help="If set, drop rows where quote_valid is False before counting")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    aspects_path = project_root / args.aspects
    reviews_path = project_root / args.reviews
    db_path = project_root / args.db
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Load
    aspects_df = pd.read_csv(aspects_path)
    reviews_df = pd.read_csv(reviews_path)
    print(f"Loaded {len(aspects_df)} aspect rows, {len(reviews_df)} reviews")

    # Optionally drop invalid quotes
    if args.require_valid_quote:
        before = len(aspects_df)
        aspects_df = aspects_df[aspects_df["quote_valid"]]
        print(f"  --require-valid-quote: kept {len(aspects_df)}/{before} rows")

    # ---- 1. aspect_counts ---------------------------------------------------
    # Dedupe at (review_id, aspect, sentiment) level - one review = one vote
    # per (aspect, sentiment), even if Haiku returned two quotes for it.
    deduped = aspects_df.drop_duplicates(
        subset=["review_id", "lounge_id", "aspect", "sentiment"]
    )
    counts = (
        deduped.groupby(["lounge_id", "aspect", "sentiment"])
        .size()
        .reset_index(name="n_reviews")
    )
    print(f"Built aspect_counts: {len(counts)} rows")

    # ---- 2. lounge_totals ---------------------------------------------------
    # total_reviews is the FULL review count for the lounge (from reviews_clean,
    # not aspects_long), so the denominator includes reviews where ABSA
    # extracted nothing.
    review_totals = (
        reviews_df.groupby("lounge_id")
        .agg(
            total_reviews=("review_id", "count"),
            mean_recency_weight=("recency_weight", "mean"),
        )
        .reset_index()
    )
    aspect_totals = (
        aspects_df.groupby("lounge_id").size().reset_index(name="total_aspect_mentions")
    )
    totals = review_totals.merge(aspect_totals, on="lounge_id", how="left")
    totals["total_aspect_mentions"] = totals["total_aspect_mentions"].fillna(0).astype(int)
    print(f"Built lounge_totals: {len(totals)} rows")

    # ---- 3. aspect_quotes (top-K per group, ranked by recency) -------------
    quotes = aspects_df.copy()
    # Stable sort by recency_weight desc, then by review_date desc as tiebreaker
    quotes["review_date"] = quotes["review_date"].astype(str)
    quotes = quotes.sort_values(
        by=["lounge_id", "aspect", "sentiment", "recency_weight", "review_date"],
        ascending=[True, True, True, False, False],
    )
    quotes["rank_in_group"] = (
        quotes.groupby(["lounge_id", "aspect", "sentiment"]).cumcount() + 1
    )
    quotes = quotes[quotes["rank_in_group"] <= args.top_quotes]
    quotes = quotes[[
        "lounge_id", "aspect", "sentiment", "quote",
        "review_id", "review_date", "recency_weight", "rank_in_group",
    ]]
    print(f"Built aspect_quotes: {len(quotes)} rows (top {args.top_quotes} per group)")

    # ---- Write to SQLite ---------------------------------------------------
    if db_path.exists():
        db_path.unlink()  # we recreate from scratch every time - cheap
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        counts.to_sql("aspect_counts", conn, if_exists="append", index=False)
        totals.to_sql("lounge_totals", conn, if_exists="append", index=False)
        quotes.to_sql("aspect_quotes", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()

    print(f"\nWrote SQLite database: {db_path}")

    # ---- Quick sanity report -----------------------------------------------
    print("\n" + "=" * 60)
    print("SANITY CHECK")
    print("=" * 60)

    print("\nTop 10 aspect_counts (by n_reviews):")
    print(counts.sort_values("n_reviews", ascending=False).head(10).to_string(index=False))

    print("\nlounge_totals:")
    print(totals.sort_values("total_reviews", ascending=False).to_string(index=False))

    print("\nSample evidence card (Tigerbay flavour_quality):")
    sample = quotes[
        (quotes["lounge_id"] == "tigerbay_kingsbury")
        & (quotes["aspect"] == "flavour_quality")
        & (quotes["sentiment"] == "positive")
    ].head(3)
    if len(sample):
        for _, row in sample.iterrows():
            print(f"  [{row['rank_in_group']}] (recency={row['recency_weight']:.2f}) "
                  f"\"{row['quote']}\"")
    else:
        print("  (no rows yet - run absa_batch first)")


if __name__ == "__main__":
    main()