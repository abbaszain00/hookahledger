# HookahLedger

A London shisha lounge intelligence engine. Takes a natural language question (e.g. "best coal management in north london") and returns a grounded recommendation backed by aspect-based sentiment analysis of thousands of real reviews.

The differentiator is specificity. Instead of "Tigerbay is great, 4.6 stars", you get "Noya leads on coal management with 11 positive reviews vs 3 negative; Tigerbay's data is thin (7 positive); Shisha Garden's negative coal mentions outnumber positive 14 to 8."

Built as the capstone for the Digital Futures Frontier AI programme, February 2026.

## How it works

Reviews from 14 London shisha lounges are scraped from Google Maps via Outscraper, cleaned, and passed through Claude Haiku for aspect-based sentiment analysis (ABSA). The output is one row per (review, aspect) pair, with sentiment and a verbatim supporting quote.

ABSA output feeds two stores:

- A SQLite database holds deterministic counts per (lounge, aspect, sentiment) and the top-ranked verified quotes per group. The LLM never invents these numbers; they are looked up at query time.
- A ChromaDB collection holds dense embeddings of context-enriched review documents, generated locally via BGE-M3.

At query time, the user's question is embedded, candidates are pulled from ChromaDB with optional metadata filters (area, price tier, aspect), aggregated per-lounge to prevent any single high-volume lounge dominating, reranked by Cohere's rerank-v3.5 with recency weighting, and the top results plus the verified quotes plus the SQLite counts are passed to Claude Sonnet for the final answer. Every double-quoted span in Sonnet's output is post-validated against the verified quote pool; mismatches are flagged with `[unverified]` markers.

## Tech stack

| Layer                | Tool                                         |
| -------------------- | -------------------------------------------- |
| Scraping             | Outscraper                                   |
| ABSA                 | Claude Haiku 4.5 (Anthropic Message Batches) |
| Embeddings           | BGE-M3 (sentence-transformers, local)        |
| Vector store         | ChromaDB                                     |
| Deterministic counts | SQLite                                       |
| Reranker             | Cohere rerank-v3.5                           |
| Generation           | Claude Sonnet 4.5                            |

## Setup

Requires Python 3.11+. Tested on Windows (Git Bash) and Linux.

```bash
git clone <repo>
cd hookahledger
python -m venv venv
source venv/Scripts/activate    # or venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

Create a `.env` file at the repo root with:

```
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...
GEMINI_API_KEY=...           # optional, only needed for legacy Gemini scripts
OPENROUTER_API_KEY=...       # optional, only needed for legacy pilots
```

The Anthropic and Cohere keys are required. Gemini and OpenRouter are only used by abandoned scripts kept for reference.

**Note on protobuf compatibility:** Set the environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before running. ChromaDB's bundled OpenTelemetry conflicts with newer protobuf versions, and this env var forces the pure-Python parser to sidestep the issue.

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # Linux/Mac/Git Bash
# or for cmd:
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## Running a query

```bash
python scripts/answer.py "best service in north london" --area "North London"
```

Filter flags:

- `--area "North London"` (also accepts Central, East, South, West London)
- `--price-tier budget|mid|premium`
- `--aspect-positive coal_management` (or any other aspect from the taxonomy)
- `--no-stream` to print the full answer at the end instead of streaming
- `--show-evidence` to print the prompt sent to Sonnet without firing the API

Inspect the corpus structure (which lounges have data on what):

```bash
python scripts/inspect_data.py
```

## Rebuilding from raw

If you have raw Outscraper CSVs in `data/raw/` and want to rebuild the entire pipeline from scratch:

```bash
# 1. Clean and join with lounge metadata
python src/clean_reviews.py --inputs data/raw/*.csv --lounges data/lounges.csv --out data/clean/reviews_clean.csv

# 2. Run ABSA via Anthropic Message Batches API (50% off, async)
python scripts/absa_batch_async.py submit
python scripts/absa_batch_async.py status     # poll until 'ended'
python scripts/absa_batch_async.py collect

# 3. Build the deterministic count store
python scripts/build_counts.py

# 4. Embed reviews into ChromaDB
python scripts/ingest_chroma_bge.py
```

The full pipeline takes about an hour on the 14-lounge / ~4,000-review corpus.

## Project structure

```
hookahledger/
  data/
    lounges.csv               14 lounges with metadata
    raw/                      Outscraper CSVs (gitignored)
    clean/
      reviews_clean.csv       3,990 cleaned reviews
      aspects_long.csv        13,414 ABSA rows
      hookahledger.sqlite     deterministic counts + verified quotes
      chroma/                 BGE-M3 vector store
  scripts/
    absa_batch_async.py       active ABSA pipeline
    build_counts.py           builds SQLite from aspects_long.csv
    ingest_chroma_bge.py      embeds reviews into ChromaDB
    retrieve.py               retrieval pipeline (filter, agg, rerank)
    answer.py                 LLM answer generation with quote validation
    inspect_data.py           corpus structural summary
  src/
    clean_reviews.py          cleaning pipeline
  requirements.txt
  README.md
```

## Known limitations

- The quote validator checks substring match against all verified quotes globally rather than per-claimed-lounge. In testing this hasn't surfaced cross-lounge misattributions, but it is a structural gap; a v2 fix would parse the claimed lounge from surrounding prose and validate against only that lounge's quote pool.
- No multi-turn conversation memory yet. Each query is independent.
- No out-of-taxonomy gate before retrieval. Off-topic queries reach the retrieval pipeline and rely on the system prompt's evidence-only rule to produce sensible "I don't have data on this" responses.
- The 14 lounges cover North, Central, East and South London but Brixton, West London, and several large Edgware Road venues are not in the dataset.

## Costs

| Provider             | Spend            | Notes                               |
| -------------------- | ---------------- | ----------------------------------- |
| Outscraper           | ~$15             | One-time scrape                     |
| Anthropic ABSA batch | $3.13            | 3,802 reviews at 50% batch discount |
| Anthropic Sonnet     | ~$0.05 per query | Roughly $1-2 for the eval set       |
| Cohere               | $0               | Free trial, 1,000 calls/month       |
| BGE-M3               | $0               | Local                               |
