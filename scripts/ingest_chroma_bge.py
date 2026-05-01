"""
ingest_chroma_bge.py - embed reviews into ChromaDB using BGE-M3 (local).

Replaces ingest_chroma.py for the BGE-M3 path.

Why a fresh script (not a flag toggle): different embedding models produce
incompatible vector dimensions (Gemini-001=3072, BGE-M3=1024). Mixing them
in one collection silently breaks similarity search. Cleaner to keep the
two paths separate, with a different default collection name (`reviews_bge`).

What this does:
  1. Loads reviews_clean.csv + aspects_long.csv
  2. Builds the same context-enriched documents as before
  3. Loads BGE-M3 once via sentence-transformers (~2 GB download first time)
  4. Encodes documents in batches on CPU (no rate limits, no API)
  5. Stores in ChromaDB collection `reviews_bge` with the same metadata
     schema as the Gemini version
  6. Runs the same 6 probe queries at the end so we can compare retrieval
     quality eyeballs vs. the Gemini ingestion

Usage:
  python scripts/ingest_chroma_bge.py                # full ingest
  python scripts/ingest_chroma_bge.py --limit 50     # smoke test
  python scripts/ingest_chroma_bge.py --rebuild      # wipe and re-embed
  python scripts/ingest_chroma_bge.py --probe-only   # eyeball existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "reviews_bge"

# Probes - same ones used in the Gemini script so we can compare like-for-like
PROBE_QUERIES = [
    "best mint flavour",
    "lounge with smooth shisha that lasts a long time",
    "rude staff and bad service",
    "value for money under 25 pounds",
    "good vibe for a date night",
    "tigerbay coal quality",
]


# ---------------------------------------------------------------------------
# Document construction (shared with Gemini script)
# ---------------------------------------------------------------------------
def build_document(review_row: dict, aspects_for_review: pd.DataFrame) -> str:
    aspect_strs = []
    if len(aspects_for_review):
        for _, ar in aspects_for_review.iterrows():
            aspect_strs.append(f"{ar['aspect']} ({ar['sentiment']})")
    aspect_block = ", ".join(aspect_strs) if aspect_strs else "none extracted"

    date_str = str(review_row.get("review_date", ""))[:10]

    return (
        f"Lounge: {review_row['lounge_name']} | "
        f"Area: {review_row['area']} | "
        f"Neighbourhood: {review_row['neighbourhood']} | "
        f"Price tier: {review_row['price_tier']} (~\u00a3{int(review_row['price_estimate_gbp'])}/head) | "
        f"Date: {date_str} | "
        f"Aspects: {aspect_block} | "
        f"Review: {review_row['review_text']}"
    )


def build_metadata(review_row: dict, aspects_for_review: pd.DataFrame) -> dict:
    aspect_keys = sorted({a for a in aspects_for_review["aspect"].tolist()})
    sentiment_pairs = sorted({
        f"{r['aspect']}__{r['sentiment']}"
        for _, r in aspects_for_review.iterrows()
    })
    return {
        "lounge_id": str(review_row["lounge_id"]),
        "lounge_name": str(review_row["lounge_name"]),
        "area": str(review_row["area"]),
        "neighbourhood": str(review_row["neighbourhood"]),
        "price_tier": str(review_row["price_tier"]),
        "price_estimate_gbp": int(review_row["price_estimate_gbp"]),
        "review_rating": int(review_row["review_rating"]) if pd.notna(review_row["review_rating"]) else 0,
        "review_date": str(review_row.get("review_date", ""))[:10],
        "recency_weight": float(review_row["recency_weight"]),
        "n_aspects": len(aspects_for_review),
        "aspects_csv": ",".join(aspect_keys) if aspect_keys else "",
        "aspect_sentiments_csv": ",".join(sentiment_pairs) if sentiment_pairs else "",
    }


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def run_probes(model: SentenceTransformer, collection) -> None:
    print("\n" + "=" * 60)
    print("RETRIEVAL SANITY CHECK (BGE-M3)")
    print("=" * 60)

    # BGE-M3 doesn't need a query instruction (the model card explicitly notes this)
    query_embeddings = model.encode(
        PROBE_QUERIES,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    for query, qvec in zip(PROBE_QUERIES, query_embeddings):
        print(f"\nQUERY: \"{query}\"")
        results = collection.query(
            query_embeddings=[qvec],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            doc = results["documents"][0][i]
            review_start = doc.find("Review: ")
            review_excerpt = doc[review_start + 8:review_start + 188] if review_start > -1 else doc[:180]
            print(f"  [{i+1}] dist={dist:.3f} | {meta['lounge_id']} ({meta['area']}) | rec={meta['recency_weight']:.2f}")
            print(f"      \"{review_excerpt}{'...' if len(doc) > review_start + 188 else ''}\"")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", default="data/clean/reviews_clean.csv")
    parser.add_argument("--aspects", default="data/clean/aspects_long.csv")
    parser.add_argument("--chroma-dir", default="data/clean/chroma")
    parser.add_argument("--collection", default=COLLECTION_NAME,
                        help="ChromaDB collection name")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Reviews per encode call (smaller = lower RAM, slower)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on total reviews to embed this run")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete the collection first and re-embed everything")
    parser.add_argument("--probe-only", action="store_true",
                        help="Skip ingestion, just run probe queries")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    chroma_dir = project_root / args.chroma_dir
    chroma_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    if args.rebuild:
        try:
            chroma_client.delete_collection(args.collection)
            print(f"Deleted existing collection '{args.collection}'")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"ChromaDB at {chroma_dir} - collection '{args.collection}' has {collection.count()} docs")

    # Load BGE-M3 (downloads ~2GB on first run; cached in ~/.cache/huggingface afterwards)
    print(f"\nLoading {MODEL_NAME}... (first time: ~2 GB download)")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    if args.probe_only:
        if collection.count() == 0:
            print("ERROR: collection is empty - run ingestion first")
            sys.exit(1)
        run_probes(model, collection)
        return

    reviews_df = pd.read_csv(project_root / args.reviews)
    aspects_df = pd.read_csv(project_root / args.aspects)
    print(f"\nLoaded {len(reviews_df)} reviews and {len(aspects_df)} aspect rows")

    aspects_by_review = {
        rid: g for rid, g in aspects_df.groupby("review_id")
    }

    # Resume support
    already_done: set[str] = set()
    if collection.count() > 0:
        offset = 0
        while True:
            chunk = collection.get(limit=5000, offset=offset, include=[])
            ids = chunk["ids"]
            if not ids:
                break
            already_done.update(ids)
            offset += len(ids)
        print(f"Already ingested: {len(already_done)} reviews - will skip them")

    todo = reviews_df[~reviews_df["review_id"].astype(str).isin(already_done)].copy()
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"Embedding {len(todo)} new reviews this run")

    if len(todo) == 0:
        print("Nothing to ingest.")
        run_probes(model, collection)
        return

    docs: list[str] = []
    metas: list[dict] = []
    ids: list[str] = []
    print("Building context-enriched documents...")
    for _, row in todo.iterrows():
        rid = str(row["review_id"])
        review_aspects = aspects_by_review.get(
            rid, pd.DataFrame(columns=aspects_df.columns)
        )
        docs.append(build_document(row.to_dict(), review_aspects))
        metas.append(build_metadata(row.to_dict(), review_aspects))
        ids.append(rid)

    # Encode in batches with progress bar. normalize_embeddings=True is the
    # BGE recommendation for cosine-similarity retrieval and gives slightly
    # better results in practice.
    n_added = 0
    with tqdm(total=len(docs), desc="Embedding", unit="doc") as bar:
        for start in range(0, len(docs), args.batch_size):
            end = start + args.batch_size
            batch_docs = docs[start:end]
            batch_metas = metas[start:end]
            batch_ids = ids[start:end]
            embeddings = model.encode(
                batch_docs,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            n_added += len(batch_ids)
            bar.update(len(batch_ids))

    print(f"\nIngested {n_added} reviews. Collection now contains {collection.count()} docs.")

    run_probes(model, collection)


if __name__ == "__main__":
    main()