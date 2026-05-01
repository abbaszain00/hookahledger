"""
test_gemini.py - one-shot connection test for Gemini embeddings.

Embeds two sample shisha review snippets and prints:
  - vector dimensions
  - cosine similarity between them (sanity check that semantically
    similar text gets similar vectors)

If this prints sensible output, the key works and we can move on
to bulk ingestion.
"""

import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


EMBEDDING_MODEL = "gemini-embedding-001"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if (norm_a and norm_b) else 0.0


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not in .env")
        sys.exit(1)
    print(f"Loaded API key: {api_key[:10]}...{api_key[-4:]}  ({len(api_key)} chars)")

    client = genai.Client(api_key=api_key)

    samples = [
        "The shisha lasted 2 hours and the coal was changed without us asking.",
        "Smooth smoke throughout, staff were on top of refreshing the coals.",
        "The food was overpriced and the chicken was undercooked.",
    ]

    print(f"\nEmbedding {len(samples)} samples with {EMBEDDING_MODEL}...")
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=samples,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
        ),
    )

    embeddings = [e.values for e in response.embeddings]
    dim = len(embeddings[0])
    print(f"  Vector dimensions: {dim}")
    print(f"  Got {len(embeddings)} embeddings\n")

    print("Sanity check - cosine similarity:")
    print(f"  [coal/duration] vs [coal/refresh]:  {cosine(embeddings[0], embeddings[1]):.3f}  (should be HIGH - both about coal)")
    print(f"  [coal/duration] vs [food complaint]: {cosine(embeddings[0], embeddings[2]):.3f}  (should be LOW - different topics)")

    print("\nGemini embeddings working.")


if __name__ == "__main__":
    main()