"""
inspect_data.py - print a structural summary of the corpus.

Reads lounges.csv + the SQLite count store and prints what's actually in
the data: lounges by area and price tier, top lounges per aspect by
positive count and recency, lounges with the most divided sentiment, etc.

Output is meant to be eyeballed - it tells us what eval queries can
defensibly expect from the system. Not a long-running tool; one-shot.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOUNGES_CSV = PROJECT_ROOT / "data" / "lounges.csv"
SQLITE_PATH = PROJECT_ROOT / "data" / "clean" / "hookahledger.sqlite"

ASPECTS = [
    "flavour_quality", "coal_management", "service_speed",
    "value_for_money", "atmosphere_vibe", "seating_comfort",
    "food_quality", "wait_time",
]


def load_lounges() -> pd.DataFrame:
    return pd.read_csv(LOUNGES_CSV)


def fetch_sqlite() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        totals = pd.read_sql("SELECT * FROM lounge_totals", conn)
        counts = pd.read_sql("SELECT * FROM aspect_counts", conn)
    finally:
        conn.close()
    return totals, counts


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    lounges = load_lounges()
    totals, counts = fetch_sqlite()

    # Merge metadata into totals for quick lookups. lounges.csv uses 'name'
    # for the lounge display name; SQLite already has lounge_name on totals.
    merged = totals.merge(
        lounges[["lounge_id", "name", "area", "neighbourhood",
                 "price_tier", "price_estimate_gbp"]].rename(
                     columns={"name": "lounge_name_csv"}),
        on="lounge_id",
        how="left",
    )

    # ---- 1. Lounges by area + price ----
    section("LOUNGES BY AREA AND PRICE TIER")
    grouped = merged.groupby(["area", "price_tier"]).agg(
        n_lounges=("lounge_id", "count"),
        lounges=("lounge_id", lambda s: ", ".join(sorted(s))),
    ).reset_index()
    print(grouped.to_string(index=False))

    # ---- 2. Lounges with full metadata ----
    section("ALL LOUNGES (with review volume and recency)")
    table = merged[[
        "lounge_id", "area", "neighbourhood", "price_tier",
        "price_estimate_gbp", "total_reviews", "mean_recency_weight",
        "total_aspect_mentions",
    ]].sort_values("total_reviews", ascending=False)
    print(table.to_string(index=False))

    # ---- 3. Top 3 lounges per aspect by positive count ----
    section("TOP 3 LOUNGES BY POSITIVE REVIEWS PER ASPECT")
    print("(by raw positive review count, then recency-weighted)\n")

    aspect_meta = counts.merge(
        merged[["lounge_id", "area", "mean_recency_weight"]],
        on="lounge_id",
        how="left",
    )

    for aspect in ASPECTS:
        pos = aspect_meta[
            (aspect_meta["aspect"] == aspect) & (aspect_meta["sentiment"] == "positive")
        ].copy()
        if pos.empty:
            print(f"\n{aspect:20s}: (no positive reviews)")
            continue
        # Recency-weighted score: positive count * mean_recency_weight
        pos["recency_score"] = pos["n_reviews"] * pos["mean_recency_weight"]
        top_raw = pos.nlargest(3, "n_reviews")
        top_recency = pos.nlargest(3, "recency_score")

        print(f"\n{aspect}")
        print(f"  by raw count:")
        for _, row in top_raw.iterrows():
            print(f"    {row['lounge_id']:25s} ({row['area']:12s}) "
                  f"= {row['n_reviews']:3d} positive (recency {row['mean_recency_weight']:.2f})")
        print(f"  by recency-weighted count:")
        for _, row in top_recency.iterrows():
            print(f"    {row['lounge_id']:25s} ({row['area']:12s}) "
                  f"= {row['recency_score']:.1f} score "
                  f"({row['n_reviews']:3d} pos x {row['mean_recency_weight']:.2f} rec)")

    # ---- 4. Most controversial lounges (high positive AND negative) ----
    section("MOST CONTROVERSIAL ASPECTS (high positive AND negative)")
    print("(lounges where one aspect has 20+ positive AND 20+ negative)\n")

    pivoted = counts.pivot_table(
        index=["lounge_id", "aspect"],
        columns="sentiment",
        values="n_reviews",
        fill_value=0,
    ).reset_index()
    if "positive" in pivoted.columns and "negative" in pivoted.columns:
        controversial = pivoted[
            (pivoted["positive"] >= 20) & (pivoted["negative"] >= 20)
        ].copy()
        controversial["ratio"] = (
            controversial["positive"] / controversial["negative"]
        ).round(2)
        controversial = controversial.sort_values(
            ["aspect", "positive"], ascending=[True, False]
        )
        if controversial.empty:
            print("  (none found)")
        else:
            print(controversial[["lounge_id", "aspect", "positive", "negative", "ratio"]]
                  .to_string(index=False))

    # ---- 5. Predominantly negative aspects per lounge ----
    section("WHERE NEGATIVE OUTWEIGHS POSITIVE")
    print("(aspects where negative > positive AND negative >= 5)\n")

    neg_dominated = pivoted[
        (pivoted.get("negative", 0) > pivoted.get("positive", 0))
        & (pivoted.get("negative", 0) >= 5)
    ].copy()
    if not neg_dominated.empty:
        neg_dominated = neg_dominated.sort_values("negative", ascending=False)
        print(neg_dominated[["lounge_id", "aspect", "positive", "negative"]]
              .to_string(index=False))

    # ---- 6. Coverage gaps - aspects with very thin data per lounge ----
    section("ASPECT COVERAGE GAPS (positive reviews <5 per aspect-lounge)")

    sparse = []
    for aspect in ASPECTS:
        for _, lounge_row in totals.iterrows():
            lid = lounge_row["lounge_id"]
            row_match = counts[
                (counts["lounge_id"] == lid)
                & (counts["aspect"] == aspect)
                & (counts["sentiment"] == "positive")
            ]
            n_pos = int(row_match["n_reviews"].iloc[0]) if not row_match.empty else 0
            if n_pos < 5:
                sparse.append((aspect, lid, n_pos))

    sparse_df = pd.DataFrame(sparse, columns=["aspect", "lounge_id", "positive"])
    if not sparse_df.empty:
        # Just summarise count by aspect
        gaps_per_aspect = sparse_df.groupby("aspect").size().sort_values(ascending=False)
        print("\n  Aspects with sparse coverage (count of lounges with <5 positives):")
        for asp, n in gaps_per_aspect.items():
            print(f"    {asp:20s} {n:2d} lounges")


if __name__ == "__main__":
    main()