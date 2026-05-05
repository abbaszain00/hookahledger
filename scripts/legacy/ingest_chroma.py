"""
ingest_chroma.py - embed reviews into ChromaDB using Gemini Embedding 001.

Pipeline:
  1. Load reviews_clean.csv + aspects_long.csv.
  2. For each review, build a context-enriched document string that includes
     lounge metadata + extracted aspects + the review text. This lets the
     embedding model "see" lounge name, area, price tier, and aspect labels
     when building the vector - improves retrieval on queries that name the
     lounge or aspect explicitly.
  3. Embed in batches of N via Gemini's embed_content (RETRIEVAL_DOCUMENT
     task type, asymmetric to the query-side encoding we'll use later).
  4. Store in a persistent ChromaDB collection with rich metadata for
     filtering at query time (lounge_id, area, price_tier, recency_weight,
     aspects_csv, etc).
  5. Resume support - the script keeps a 'review_id -> embedded' set in the
     ChromaDB collection itself, so re-running skips already-ingested docs.
  6. Sanity check at the end: run a few canned queries and print the top
     hits so we can eyeball that retrieval is working before moving on.

Usage:
  python scripts/ingest_chroma.py                   # full ingest
  python scripts/ingest_chroma.py --limit 200       # smoke test
  python scripts/ingest_chroma.py --rebuild         # wipe and re-embed
  python scripts/ingest_chroma.py --probe-only      # skip ingest, run probes
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.config import Settings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm


EMBEDDING_MODEL = "gemini-embedding-001"
COLLECTION_NAME = "reviews"

# Canned probes - we run these after ingestion to sanity-check retrieval.
# Each is a (query, expected behaviour) pair we can eyeball.
PROBE_QUERIES = [
    "best mint flavour",
    "lounge with smooth shisha that lasts a long time",  # implicit coal_management
    "rude staff and bad service",
    "value for money under 25 pounds",
    "good vibe for a date night",
    "tigerbay coal quality",  # lounge-named query
]


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------
def build_document(review_row: dict, aspects_for_review: pd.DataFrame) -> str:
    """Build the context-enriched string that goes into the embedding model.

    Format chosen so the model 'sees' lounge name, area, price tier, and
    aspect labels alongside the review text. Bar-separated for readability;
    the embedding model doesn't care about format, but it does help anyone
    debugging retrieval results later.
    """
    aspect_strs = []
    if len(aspects_for_review):
        for _, ar in aspects_for_review.iterrows():
            aspect_strs.append(f"{ar['aspect']} ({ar['sentiment']})")
    aspect_block = ", ".join(aspect_strs) if aspect_strs else "none extracted"

    date_str = str(review_row.get("review_date", ""))[:10]  # YYYY-MM-DD only

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
    """Per-document metadata for ChromaDB - used for filtering at query time.

    Note: Chroma metadata only allows scalars and arrays. We store the list of
    aspects as a comma-joined string so we can use $contains filtering later.
    """
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
        # Comma-joined strings for $contains filtering
        "aspects_csv": ",".join(aspect_keys) if aspect_keys else "",
        "aspect_sentiments_csv": ",".join(sentiment_pairs) if sentiment_pairs else "",
    }


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_batch(client: genai.Client, texts: list[str], max_retries: int = 6) -> list[list[float]]:
    """Embed a batch of texts via Gemini. Retries on transient errors.

    On 429 RESOURCE_EXHAUSTED, parses the retry-after hint from the error
    message and sleeps for that long (with a small buffer) instead of
    using fixed exponential backoff.
    """
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            last_err = e
            err_str = str(e)
            # Parse "Please retry in NNs" hint from 429 errors
            sleep_for = (2 ** (attempt + 1)) + (attempt * 0.5)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
                if m:
                    sleep_for = float(m.group(1)) + 2.0  # buffer
                else:
                    sleep_for = 60.0  # safe default for free-tier RPM resets
                print(f"  [rate-limited, sleeping {sleep_for:.0f}s before retry]")
            time.sleep(sleep_for)
    raise last_err if last_err else RuntimeError("embed_batch: no error captured")


def embed_query(client: genai.Client, text: str) -> list[float]:
    """Embed a SINGLE query string with the RETRIEVAL_QUERY task type.

    Asymmetric retrieval: documents use RETRIEVAL_DOCUMENT, queries use
    RETRIEVAL_QUERY. Gemini packs them into compatible spaces but optimises
    each for its role.
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return response.embeddings[0].values


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def run_probes(genai_client: genai.Client, collection) -> None:
    print("\n" + "=" * 60)
    print("RETRIEVAL SANITY CHECK")
    print("=" * 60)
    for query in PROBE_QUERIES:
        print(f"\nQUERY: \"{query}\"")
        try:
            qvec = embed_query(genai_client, query)
        except Exception as e:
            print(f"  embed_query failed: {e}")
            continue
        results = collection.query(
            query_embeddings=[qvec],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            doc = results["documents"][0][i]
            # Trim the doc - the prefix is metadata noise, the review starts at "Review:"
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
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Reviews per Gemini embed call (Gemini caps at 100/batch)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on total reviews to embed this run")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete the collection first and re-embed everything")
    parser.add_argument("--probe-only", action="store_true",
                        help="Skip ingestion, just run probe queries against the existing collection")
    parser.add_argument("--rpm-cap", type=int, default=90,
                        help="Pace requests under this requests-per-minute (default 90 to stay under free tier's 100)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not in .env")
        sys.exit(1)

    chroma_dir = project_root / args.chroma_dir
    chroma_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    genai_client = genai.Client(api_key=api_key)

    if args.rebuild:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"ChromaDB at {chroma_dir} - collection '{COLLECTION_NAME}' has {collection.count()} docs")

    if args.probe_only:
        if collection.count() == 0:
            print("ERROR: collection is empty - run ingestion first")
            sys.exit(1)
        run_probes(genai_client, collection)
        return

    # Load data
    reviews_df = pd.read_csv(project_root / args.reviews)
    aspects_df = pd.read_csv(project_root / args.aspects)
    print(f"Loaded {len(reviews_df)} reviews and {len(aspects_df)} aspect rows")

    # Group aspects by review_id for fast lookup
    aspects_by_review = {
        rid: g for rid, g in aspects_df.groupby("review_id")
    }

    # Resume support: skip review_ids already in the collection
    already_done: set[str] = set()
    if collection.count() > 0:
        # Pull existing IDs in chunks of 5000
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
        run_probes(genai_client, collection)
        return

    # Build documents and metadata up front - cheap, all-in-memory
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

    # Embed and add in batches, paced to stay under the RPM cap.
    # min_seconds_between_batches = 60 / rpm_cap, so 90 RPM -> 0.667s gap.
    min_gap = 60.0 / max(args.rpm_cap, 1)
    n_added = 0
    last_call_at = 0.0
    print(f"\nPacing: <= {args.rpm_cap} requests/min (gap >= {min_gap:.2f}s between batches)")
    with tqdm(total=len(docs), desc="Embedding", unit="doc") as bar:
        for start in range(0, len(docs), args.batch_size):
            # Pace - sleep just enough to stay under cap
            elapsed = time.time() - last_call_at
            if elapsed < min_gap:
                time.sleep(min_gap - elapsed)
            end = start + args.batch_size
            batch_docs = docs[start:end]
            batch_metas = metas[start:end]
            batch_ids = ids[start:end]
            last_call_at = time.time()
            embeddings = embed_batch(genai_client, batch_docs)
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            n_added += len(batch_ids)
            bar.update(len(batch_ids))

    print(f"\nIngested {n_added} reviews. Collection now contains {collection.count()} docs.")

    run_probes(genai_client, collection)


if __name__ == "__main__":
    main()