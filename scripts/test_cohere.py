"""
test_cohere.py - one-shot connection test for Cohere Rerank.

Sends a query and 4 candidate documents to the rerank-v3.5 model
and prints the relevance scores. Two of the candidates are obviously
relevant, two are obviously not - if scores rank them correctly the
key works and rerank is doing what we expect.
"""

import os
import sys
from pathlib import Path

import cohere
from dotenv import load_dotenv


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("ERROR: COHERE_API_KEY not in .env")
        sys.exit(1)
    print(f"Loaded API key: {api_key[:8]}...{api_key[-4:]}  ({len(api_key)} chars)")

    client = cohere.ClientV2(api_key=api_key)

    query = "best coal management at a shisha lounge"

    candidates = [
        # Obviously relevant - explicit coal management complaint
        "Sheesha coal just kept bouncing around creating a harsh taste as they don't use any hmd. Used to be good, gone downhill.",
        # Obviously irrelevant - about food, no shisha mention
        "The chicken karaage was perfectly crispy and the lamb cutlets were juicy.",
        # Genuinely relevant - implicit coal language
        "Shisha lasted nearly 2 hours and the heat stayed strong throughout. Coals were swapped twice without us asking.",
        # Generic positive, weak signal
        "The best place to be, lovely staff, will come again.",
    ]

    print(f"\nQuery: \"{query}\"")
    print(f"Reranking {len(candidates)} candidates with rerank-v3.5...\n")

    response = client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=candidates,
        top_n=len(candidates),  # return all so we can see all scores
    )

    print("Reranked (in order of relevance):")
    for r in response.results:
        idx = r.index
        score = r.relevance_score
        excerpt = candidates[idx][:90] + ("..." if len(candidates[idx]) > 90 else "")
        print(f"  [{idx}] score={score:.3f}  {excerpt}")

    print("\nExpected ordering: explicit-coal review highest, food review lowest.")
    print("Cohere rerank working." if response.results else "")


if __name__ == "__main__":
    main()