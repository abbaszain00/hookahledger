"""
retrieve.py - the retrieval pipeline for HookahLedger.

Implements steps 3-5 of the design doc retrieval flow, plus a poor-man's
recency-weighted rerank as a stand-in for Cohere (step 6) until we plug
that in.

What it does end-to-end:

  query (str)
    -> [4] metadata pre-filter using Chroma `where` clauses
    -> [5] per-lounge aggregation: pull top-3 chunks per lounge, then merge
           across lounges to a fixed candidate pool
    -> [6'] recency-weighted rescore: combined = (1 - dist) * recency_weight
            top-K returned
    -> [3] for each surviving lounge_id, fetch deterministic counts from
           SQLite (lounge_totals + relevant aspect_counts)

Output is a dataclass-like dict structure that's directly suitable as the
"evidence" payload to hand to the LLM later. Designed so the next layer
(Sonnet answer generation) just receives this and writes the prose.

Usage:
  CLI:  python scripts/retrieve.py "best coal in north london"
        python scripts/retrieve.py "good vibes for a date" --area "Central London"

  Module: from scripts.retrieve import RetrievalPipeline
          pipeline = RetrievalPipeline()
          result = pipeline.retrieve("best mint flavour", area="North London")

Filters supported (CLI flags or kwargs):
  --area "North London"           - filter to one area
  --price-tier premium            - filter to one price tier (budget/mid/premium)
  --aspect-positive coal_management - filter to lounges with at least one
                                       review where this aspect is positive
  --top-k 5                       - final result count
  --candidates 30                 - candidate pool size before reranking
  --per-lounge 3                  - chunks pulled per lounge in aggregation
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    import cohere
    from dotenv import load_dotenv
    _HAS_COHERE = True
except ImportError:
    _HAS_COHERE = False


# ---------------------------------------------------------------------------
# Defaults - all paths relative to project root
# ---------------------------------------------------------------------------
DEFAULT_CHROMA_DIR = "data/clean/chroma"
DEFAULT_COLLECTION = "reviews_bge"
DEFAULT_SQLITE = "data/clean/hookahledger.sqlite"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"

# Knobs - exposed via CLI but defaulted here
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATES = 30
DEFAULT_PER_LOUNGE = 3
# Recency weighting: score = sim * (floor + (1-floor) * recency_weight).
# 1.0 = ignore recency entirely. 0.7 = a 0-recency review keeps 70% of its
# similarity. 0.5 = was the original aggressive setting (made stale reviews
# half-weight). 0.7 is a reasonable middle ground - new reviews still
# preferred for ties, but ancient genuine matches aren't crushed.
DEFAULT_RECENCY_FLOOR = 0.7
# When the Cohere reranker is doing the semantic work, recency only needs to
# break ties. So a much gentler floor.
DEFAULT_RECENCY_FLOOR_WITH_RERANK = 0.85

DEFAULT_RERANK_MODEL = "rerank-v3.5"
# Pool sent to the reranker AFTER per-lounge aggregation. Cohere accepts up
# to 1000 docs but smaller is faster and conserves the free trial quota.
DEFAULT_RERANK_POOL = 25


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    """One review chunk surviving the full retrieval pipeline."""
    review_id: str
    lounge_id: str
    lounge_name: str
    area: str
    neighbourhood: str
    price_tier: str
    review_date: str
    recency_weight: float
    document: str
    distance: float          # raw cosine distance from Chroma (0=identical)
    similarity: float        # 1 - distance, easier to read
    score: float             # final reranked score
    cohere_relevance: float | None  # rerank-v3.5 score, None if rerank skipped
    aspects_csv: str
    aspect_sentiments_csv: str


@dataclass
class LoungeEvidence:
    """Per-lounge evidence pack: counts, totals, and the chunks that survived."""
    lounge_id: str
    lounge_name: str
    area: str
    total_reviews: int
    total_aspect_mentions: int
    mean_recency_weight: float
    aspect_counts: list[dict]  # rows from aspect_counts table
    chunks: list[RetrievedChunk]


@dataclass
class RetrievalResult:
    query: str
    filters: dict
    candidates_pulled: int
    chunks: list[RetrievedChunk]
    lounges: list[LoungeEvidence]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class RetrievalPipeline:
    def __init__(
        self,
        project_root: Path | None = None,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        collection: str = DEFAULT_COLLECTION,
        sqlite_path: str = DEFAULT_SQLITE,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        chroma_full = self.project_root / chroma_dir
        sqlite_full = self.project_root / sqlite_path

        if not chroma_full.exists():
            raise FileNotFoundError(f"ChromaDB not found at {chroma_full}")
        if not sqlite_full.exists():
            raise FileNotFoundError(f"SQLite not found at {sqlite_full}")

        chroma_client = chromadb.PersistentClient(
            path=str(chroma_full),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = chroma_client.get_collection(collection)
        self.sqlite_path = sqlite_full

        # Sentence transformer is reused across queries
        print(f"Loading embedding model {embed_model}...", file=sys.stderr)
        self.embedder = SentenceTransformer(embed_model)

        # Cohere is optional. If the key isn't set, retrieve() can still run
        # with rerank=False. We don't error here - we error at call time.
        self._cohere_client: cohere.ClientV2 | None = None
        if _HAS_COHERE:
            load_dotenv(self.project_root / ".env")
            api_key = os.getenv("COHERE_API_KEY")
            if api_key:
                self._cohere_client = cohere.ClientV2(api_key=api_key)

    # -------- Step 4: build the Chroma `where` filter -----------------------
    def _build_where(
        self,
        area: str | None,
        price_tier: str | None,
        aspect_positive: str | None,
    ) -> dict | None:
        """Compose Chroma's $and filter. Returns None if no constraints set."""
        clauses: list[dict] = []
        if area:
            clauses.append({"area": {"$eq": area}})
        if price_tier:
            clauses.append({"price_tier": {"$eq": price_tier}})
        if aspect_positive:
            # aspect_sentiments_csv looks like 'coal_management__positive,flavour_quality__negative'.
            # Chroma's where_document supports $contains on the document body, but our
            # aspects field is metadata. Chroma's metadata filter has an $eq but no
            # substring op. Workaround: use where_document on the *Aspects:* line
            # in the document text - we know the format because we built the docs
            # ourselves in build_document. For now we only filter by aspect via the
            # document text in step 5, see _query_chroma. Skip metadata filter here.
            pass
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    # -------- Step 5a: dense candidate pull --------------------------------
    def _query_chroma(
        self,
        query_vec: list[float],
        where: dict | None,
        n_candidates: int,
        aspect_positive: str | None,
    ) -> dict:
        """Pull a wide candidate pool from Chroma. We over-fetch so per-lounge
        aggregation has options to reshuffle.

        If aspect_positive is supplied we additionally use Chroma's where_document
        contains-filter to require the literal `<aspect>__positive` token in the
        document text (which build_document includes in the Aspects: section
        as e.g. 'coal_management (positive)' - we filter on the substring
        '(positive)' adjacent to the aspect name).
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vec],
            "n_results": n_candidates,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        if aspect_positive is not None:
            # build_document writes 'coal_management (positive)' into the doc
            kwargs["where_document"] = {
                "$contains": f"{aspect_positive} (positive)"
            }
        return self.collection.query(**kwargs)

    # -------- Step 5b: per-lounge aggregation ------------------------------
    @staticmethod
    def _aggregate_per_lounge(
        raw_results: dict,
        per_lounge_cap: int,
        recency_floor: float,
    ) -> list[RetrievedChunk]:
        """Deduplicate so no single lounge dominates the candidate pool.
        Keep at most `per_lounge_cap` chunks per lounge. Preserves the order
        in which Chroma returned them, so within-lounge the best hits stay.

        Recency reweighting: score = similarity * (recency_floor + (1-recency_floor) * recency_weight).
        With recency_floor=1.0 recency is ignored (pure similarity).
        With recency_floor=0.5 a 0-recency review keeps half its similarity.
        With recency_floor=0.0 ancient reviews go to zero.
        """
        ids = raw_results["ids"][0]
        docs = raw_results["documents"][0]
        metas = raw_results["metadatas"][0]
        dists = raw_results["distances"][0]

        per_lounge: dict[str, list[RetrievedChunk]] = {}
        for review_id, doc, meta, dist in zip(ids, docs, metas, dists):
            lounge_id = meta.get("lounge_id", "unknown")
            bucket = per_lounge.setdefault(lounge_id, [])
            if len(bucket) >= per_lounge_cap:
                continue
            similarity = 1.0 - float(dist)
            recency = float(meta.get("recency_weight", 0.0))
            recency_factor = recency_floor + (1.0 - recency_floor) * recency
            bucket.append(RetrievedChunk(
                review_id=str(review_id),
                lounge_id=lounge_id,
                lounge_name=str(meta.get("lounge_name", "")),
                area=str(meta.get("area", "")),
                neighbourhood=str(meta.get("neighbourhood", "")),
                price_tier=str(meta.get("price_tier", "")),
                review_date=str(meta.get("review_date", "")),
                recency_weight=recency,
                document=str(doc),
                distance=float(dist),
                similarity=similarity,
                # poor-man's rerank: similarity gently nudged by recency. Cohere
                # may overwrite this score if rerank is on (see _rerank).
                score=similarity * recency_factor,
                cohere_relevance=None,
                aspects_csv=str(meta.get("aspects_csv", "")),
                aspect_sentiments_csv=str(meta.get("aspect_sentiments_csv", "")),
            ))

        # Flatten and sort by final score desc
        flat = [c for bucket in per_lounge.values() for c in bucket]
        flat.sort(key=lambda c: c.score, reverse=True)
        return flat

    # -------- Step 6: Cohere rerank ----------------------------------------

    def _rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        recency_floor: float,
        rerank_model: str,
    ) -> list[RetrievedChunk]:
        """Send a pool of candidate chunks to Cohere's reranker, then rescore.

        Final score: cohere_relevance * (recency_floor + (1-recency_floor) * recency).
        Recency only nudges - the reranker's relevance is the dominant signal.

        On Cohere failure (429, 5xx, network), falls back to similarity *
        recency ordering using DEFAULT_RECENCY_FLOOR (the no-rerank floor),
        logs a warning to stderr, and returns the chunks. The query still
        completes; the user just gets a degraded ranking instead of a crash.
        """
        if not chunks:
            return chunks
        if self._cohere_client is None:
            raise RuntimeError(
                "Cohere client not initialised. Either install cohere and "
                "set COHERE_API_KEY, or call retrieve(rerank=False)."
            )

        # We send the document body. Could send a stripped version (just the
        # 'Review:' suffix) but the metadata prefix actually helps the
        # reranker make use of lounge name / area when the query mentions them.
        documents = [c.document for c in chunks]
        try:
            response = self._cohere_client.rerank(
                model=rerank_model,
                query=query,
                documents=documents,
                top_n=len(documents),
            )
        except Exception as e:
            print(
                f"[retrieve] Cohere rerank failed ({type(e).__name__}: {e}). "
                f"Falling back to similarity-weighted ordering.",
                file=sys.stderr,
            )
            # Rescore with the no-rerank recency floor. Chunks already have
            # similarity computed; just recompute score with the harsher floor
            # since we no longer have the reranker doing the heavy lifting.
            fallback_floor = DEFAULT_RECENCY_FLOOR
            fallback: list[RetrievedChunk] = []
            for c in chunks:
                recency_factor = (
                    fallback_floor + (1.0 - fallback_floor) * c.recency_weight
                )
                fallback.append(RetrievedChunk(
                    review_id=c.review_id,
                    lounge_id=c.lounge_id,
                    lounge_name=c.lounge_name,
                    area=c.area,
                    neighbourhood=c.neighbourhood,
                    price_tier=c.price_tier,
                    review_date=c.review_date,
                    recency_weight=c.recency_weight,
                    document=c.document,
                    distance=c.distance,
                    similarity=c.similarity,
                    score=c.similarity * recency_factor,
                    cohere_relevance=None,
                    aspects_csv=c.aspects_csv,
                    aspect_sentiments_csv=c.aspect_sentiments_csv,
                ))
            fallback.sort(key=lambda c: c.score, reverse=True)
            return fallback

        # Cohere returns results in ranked order with `index` pointing back
        # to the original list. Rebuild the list in that order, attaching
        # the relevance score and the recency-nudged final score.
        reranked: list[RetrievedChunk] = []
        for r in response.results:
            c = chunks[r.index]
            recency_factor = recency_floor + (1.0 - recency_floor) * c.recency_weight
            reranked.append(RetrievedChunk(
                review_id=c.review_id,
                lounge_id=c.lounge_id,
                lounge_name=c.lounge_name,
                area=c.area,
                neighbourhood=c.neighbourhood,
                price_tier=c.price_tier,
                review_date=c.review_date,
                recency_weight=c.recency_weight,
                document=c.document,
                distance=c.distance,
                similarity=c.similarity,
                score=float(r.relevance_score) * recency_factor,
                cohere_relevance=float(r.relevance_score),
                aspects_csv=c.aspects_csv,
                aspect_sentiments_csv=c.aspect_sentiments_csv,
            ))
        # Sort by the new score (in case the recency nudge reordered things)
        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked


    # -------- Step 3: SQLite counts ----------------------------------------
    def _fetch_counts(self, lounge_ids: list[str]) -> dict[str, dict]:
        """Returns {lounge_id: {totals: row, aspect_counts: [rows]}} dict.
        Counts come from the SQLite count store, NEVER from the LLM.
        """
        if not lounge_ids:
            return {}
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" for _ in lounge_ids)
            totals_rows = conn.execute(
                f"SELECT * FROM lounge_totals WHERE lounge_id IN ({placeholders})",
                lounge_ids,
            ).fetchall()
            counts_rows = conn.execute(
                f"SELECT * FROM aspect_counts WHERE lounge_id IN ({placeholders})",
                lounge_ids,
            ).fetchall()
        finally:
            conn.close()

        result: dict[str, dict] = {lid: {"totals": None, "aspect_counts": []} for lid in lounge_ids}
        for row in totals_rows:
            result[row["lounge_id"]]["totals"] = dict(row)
        for row in counts_rows:
            result[row["lounge_id"]]["aspect_counts"].append(dict(row))
        return result

    def fetch_verified_quotes(
        self,
        lounge_ids: list[str],
        per_lounge_limit: int = 12,
    ) -> dict[str, list[dict]]:
        """Pull pre-validated quotes from the SQLite aspect_quotes table.

        These are guaranteed to be exact substrings of real reviews (the
        ABSA pipeline ran a substring check at extraction time), so they
        are safe to hand to the LLM as quotable content.

        Returns {lounge_id: [quote_row, ...]} where each quote_row is a
        dict with keys aspect, sentiment, quote, review_date,
        recency_weight, rank_in_group.

        We pull rank_in_group <= 3 (the top 3 quotes per aspect/sentiment)
        and cap each lounge at per_lounge_limit total to keep the prompt
        manageable.
        """
        if not lounge_ids:
            return {}
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" for _ in lounge_ids)
            rows = conn.execute(
                f"""
                SELECT lounge_id, aspect, sentiment, quote, review_date,
                       recency_weight, rank_in_group
                FROM aspect_quotes
                WHERE lounge_id IN ({placeholders})
                ORDER BY lounge_id, recency_weight DESC, rank_in_group ASC
                """,
                lounge_ids,
            ).fetchall()
        finally:
            conn.close()

        result: dict[str, list[dict]] = {lid: [] for lid in lounge_ids}
        for row in rows:
            lid = row["lounge_id"]
            if len(result[lid]) >= per_lounge_limit:
                continue
            result[lid].append(dict(row))
        return result

    # -------- Public entry point -------------------------------------------
    def retrieve(
        self,
        query: str,
        area: str | None = None,
        price_tier: str | None = None,
        aspect_positive: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        candidates: int = DEFAULT_CANDIDATES,
        per_lounge: int = DEFAULT_PER_LOUNGE,
        rerank: bool = True,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_pool: int = DEFAULT_RERANK_POOL,
        recency_floor: float | None = None,
    ) -> RetrievalResult:
        # Choose recency floor based on whether rerank is on. Caller can override.
        if recency_floor is None:
            recency_floor = (
                DEFAULT_RECENCY_FLOOR_WITH_RERANK if rerank else DEFAULT_RECENCY_FLOOR
            )

        # Embed the query (BGE-M3 doesn't need a query instruction)
        query_vec = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

        # Step 4: build metadata filter
        where = self._build_where(area, price_tier, aspect_positive)

        # Step 5: pull candidates and aggregate
        raw = self._query_chroma(query_vec, where, candidates, aspect_positive)
        if not raw["ids"] or not raw["ids"][0]:
            return RetrievalResult(
                query=query,
                filters={"area": area, "price_tier": price_tier, "aspect_positive": aspect_positive},
                candidates_pulled=0,
                chunks=[],
                lounges=[],
            )

        candidates_pulled = len(raw["ids"][0])
        # When rerank is on, pre-rerank scoring should NOT apply recency yet -
        # we want raw similarity ordering for picking the rerank pool, then
        # apply recency post-rerank in _rerank itself.
        agg_recency_floor = 1.0 if rerank else recency_floor
        aggregated = self._aggregate_per_lounge(raw, per_lounge, agg_recency_floor)

        # Step 6: rerank (or fall back to similarity ordering)
        if rerank:
            pool = aggregated[:rerank_pool]
            reranked = self._rerank(query, pool, recency_floor, rerank_model)
            top_chunks = reranked[:top_k]
        else:
            top_chunks = aggregated[:top_k]

        # Step 3: pull counts for surviving lounges
        unique_lounges = list({c.lounge_id for c in top_chunks})
        counts_by_lounge = self._fetch_counts(unique_lounges)

        # Bundle into per-lounge evidence packs
        lounges = []
        for lid in unique_lounges:
            chunks_for_lounge = [c for c in top_chunks if c.lounge_id == lid]
            counts_data = counts_by_lounge.get(lid, {})
            totals = counts_data.get("totals") or {}
            lounges.append(LoungeEvidence(
                lounge_id=lid,
                lounge_name=chunks_for_lounge[0].lounge_name if chunks_for_lounge else "",
                area=chunks_for_lounge[0].area if chunks_for_lounge else "",
                total_reviews=int(totals.get("total_reviews", 0)),
                total_aspect_mentions=int(totals.get("total_aspect_mentions", 0)),
                mean_recency_weight=float(totals.get("mean_recency_weight", 0.0)),
                aspect_counts=counts_data.get("aspect_counts", []),
                chunks=chunks_for_lounge,
            ))
        # Sort lounges by their best chunk's score
        lounges.sort(key=lambda lg: max(c.score for c in lg.chunks) if lg.chunks else 0, reverse=True)

        return RetrievalResult(
            query=query,
            filters={"area": area, "price_tier": price_tier, "aspect_positive": aspect_positive},
            candidates_pulled=candidates_pulled,
            chunks=top_chunks,
            lounges=lounges,
        )


# ---------------------------------------------------------------------------
# Pretty-print for CLI use
# ---------------------------------------------------------------------------
def render_result(result: RetrievalResult) -> str:
    out: list[str] = []
    out.append(f"Query: \"{result.query}\"")
    f = result.filters
    if any(f.values()):
        out.append(f"Filters: {', '.join(f'{k}={v}' for k, v in f.items() if v)}")
    out.append(f"Candidate pool: {result.candidates_pulled} chunks pulled, "
               f"{len(result.chunks)} returned across {len(result.lounges)} lounges")

    for i, lg in enumerate(result.lounges, 1):
        out.append("")
        out.append(f"=== [{i}] {lg.lounge_name} ({lg.area}) ===")
        out.append(f"  Total reviews: {lg.total_reviews} | "
                   f"aspect mentions: {lg.total_aspect_mentions} | "
                   f"mean recency: {lg.mean_recency_weight:.2f}")

        # Show the top 3 aspect_counts with the highest n_reviews
        if lg.aspect_counts:
            top_aspects = sorted(
                lg.aspect_counts, key=lambda r: r["n_reviews"], reverse=True
            )[:5]
            asp_strs = [
                f"{r['aspect']} {r['sentiment']}={r['n_reviews']}" for r in top_aspects
            ]
            out.append(f"  Top aspects: {' | '.join(asp_strs)}")

        for j, c in enumerate(lg.chunks, 1):
            review_start = c.document.find("Review: ")
            excerpt = c.document[review_start + 8:] if review_start > -1 else c.document
            excerpt = excerpt[:240] + ("..." if len(excerpt) > 240 else "")
            if c.cohere_relevance is not None:
                score_line = (
                    f"  [{i}.{j}] score={c.score:.3f} cohere={c.cohere_relevance:.3f} "
                    f"sim={c.similarity:.3f} rec={c.recency_weight:.2f} | {c.review_date}"
                )
            else:
                score_line = (
                    f"  [{i}.{j}] score={c.score:.3f} sim={c.similarity:.3f} "
                    f"rec={c.recency_weight:.2f} | {c.review_date}"
                )
            out.append(score_line)
            out.append(f"        \"{excerpt}\"")

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--area", default=None,
                        help="Filter by area, e.g. 'North London'")
    parser.add_argument("--price-tier", default=None, choices=["budget", "mid", "premium"],
                        help="Filter by price tier")
    parser.add_argument("--aspect-positive", default=None,
                        help="Require lounges to have this aspect rated positive in at "
                             "least one review (e.g. coal_management)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--per-lounge", type=int, default=DEFAULT_PER_LOUNGE)
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip Cohere reranker (use similarity*recency only)")
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL,
                        help="Cohere rerank model name")
    parser.add_argument("--rerank-pool", type=int, default=DEFAULT_RERANK_POOL,
                        help="Number of post-aggregation candidates sent to reranker")
    parser.add_argument("--recency-floor", type=float, default=None,
                        help="Override recency weighting floor (0=harsh, 1=ignored). "
                             "Default 0.85 with rerank, 0.7 without.")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON instead of pretty output")
    args = parser.parse_args()

    pipeline = RetrievalPipeline()
    result = pipeline.retrieve(
        query=args.query,
        area=args.area,
        price_tier=args.price_tier,
        aspect_positive=args.aspect_positive,
        top_k=args.top_k,
        candidates=args.candidates,
        per_lounge=args.per_lounge,
        rerank=not args.no_rerank,
        rerank_model=args.rerank_model,
        rerank_pool=args.rerank_pool,
        recency_floor=args.recency_floor,
    )

    if args.json:
        # Convert dataclasses to dicts
        payload = {
            "query": result.query,
            "filters": result.filters,
            "candidates_pulled": result.candidates_pulled,
            "chunks": [asdict(c) for c in result.chunks],
            "lounges": [
                {
                    **{k: v for k, v in asdict(lg).items() if k != "chunks"},
                    "chunks": [asdict(c) for c in lg.chunks],
                }
                for lg in result.lounges
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_result(result))


if __name__ == "__main__":
    main()