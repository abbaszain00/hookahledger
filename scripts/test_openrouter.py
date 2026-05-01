"""
test_openrouter.py - one-shot connection test for OpenRouter via Haiku.

Fires a single review through Haiku and asks it to extract one aspect.
Goal: confirm the key works, the model responds, and JSON parsing works.
NOT representative of the real ABSA prompt - just a smoke test.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    # Load .env from the project root (this script lives in scripts/)
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in .env")
        sys.exit(1)
    print(f"Loaded API key: {api_key[:12]}...{api_key[-4:]}  ({len(api_key)} chars)")

    # OpenRouter is OpenAI-compatible - just point base_url at it
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    test_review = (
        "Mint flavour was incredible but we waited 40 minutes for coal. "
        "Staff were friendly though. £25 a head felt fair."
    )

    print(f"\nSending test review:\n  {test_review}\n")

    resp = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[
            {
                "role": "system",
                "content": (
                    "You read shisha lounge reviews and pull out one aspect-sentiment pair. "
                    "Reply ONLY with valid JSON in this shape: "
                    '{"aspect": "...", "sentiment": "positive|negative|mixed", "quote": "..."}'
                ),
            },
            {"role": "user", "content": test_review},
        ],
        temperature=0,
        max_tokens=200,
    )

    raw = resp.choices[0].message.content
    print(f"Raw response:\n  {raw}\n")

    # Try to parse it as JSON
    try:
        parsed = json.loads(raw)
        print("Parsed JSON:")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as e:
        # Sometimes models wrap JSON in ```json fences - strip them and retry
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(cleaned)
            print("Parsed JSON (after stripping fences):")
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print(f"Could not parse as JSON: {e}")
            sys.exit(1)

    # Token usage report
    if resp.usage:
        print(f"\nToken usage: {resp.usage.prompt_tokens} in, {resp.usage.completion_tokens} out")
        print(f"Model used: {resp.model}")

    print("\nOpenRouter connection works.")


if __name__ == "__main__":
    main()