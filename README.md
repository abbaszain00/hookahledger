# HookahLedger

A London shisha lounge intelligence engine. Takes a natural language question (e.g. "best coal management in north london") and returns a grounded recommendation backed by aspect-based sentiment analysis of thousands of real reviews.

The differentiator is specificity. Instead of "Tigerbay is great, 4.6 stars", you get "Noya leads on coal management with 11 positive reviews vs 3 negative; Tigerbay's data is thin (7 positive); Shisha Garden's negative coal mentions outnumber positive 14 to 8."

Built as the capstone for the Digital Futures Frontier AI programme, May 2026.

## How it works

Reviews from 14 London shisha lounges are scraped from Google Maps via Outscraper, cleaned, and passed through Claude Haiku for aspect-based sentiment analysis (ABSA). The output is one row per (review, aspect) pair, with sentiment and a verbatim supporting quote.

ABSA output feeds two stores:

- A SQLite database holds deterministic counts per (lounge, aspect, sentiment) and the top-ranked verified quotes per group. The LLM never invents these numbers; they are looked up at query time.
- A ChromaDB collection holds dense embeddings of context-enriched review documents, generated locally via BGE-M3.

At query time, the user's question first passes through a LangGraph query understanding agent that extracts structured filters (area, price tier, aspect) and detects out-of-taxonomy questions. The cleaned query is embedded, candidates are pulled from ChromaDB with the inferred metadata filters, aggregated per-lounge to prevent any single high-volume lounge dominating, reranked by Cohere's rerank-v3.5 with recency weighting, and the top results plus the verified quotes plus the SQLite counts are passed to Claude Sonnet for the final answer. Every double-quoted span in Sonnet's output is post-validated against the verified quote pool; mismatches are flagged with `[unverified]` markers.

The system supports multi-turn conversations: filters from earlier turns are inherited and surfaced as "carried forward" pills in the UI, and a user can pivot focus to a single lounge mid-conversation.

## Tech stack

| Layer                | Tool                                         |
| -------------------- | -------------------------------------------- |
| Scraping             | Outscraper                                   |
| ABSA                 | Claude Haiku 4.5 (Anthropic Message Batches) |
| Embeddings           | BGE-M3 (sentence-transformers, local)        |
| Vector store         | ChromaDB                                     |
| Deterministic counts | SQLite                                       |
| Query understanding  | LangGraph agent (Claude Haiku 4.5)           |
| Reranker             | Cohere rerank-v3.5                           |
| Generation           | Claude Sonnet 4.5                            |
| Backend              | FastAPI with SSE streaming                   |
| Frontend             | React + Vite                                 |

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
```

Both keys are required for the live pipeline.

**Note on protobuf compatibility:** ChromaDB's bundled OpenTelemetry conflicts with newer protobuf versions. The backend (`app/main.py`) and agent modules set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` in code, so running the FastAPI server works out of the box. If you run scripts directly (e.g. `python scripts/retrieve.py`), set the env var first:

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # Linux/Mac/Git Bash
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python      # cmd
```

## Running a query

CLI:

```bash
python scripts/answer.py "best service in north london" --area "North London"
```

Filter flags:

- `--area "North London"` (also accepts Central, East, South, West London)
- `--price-tier budget|mid|premium`
- `--aspect-positive coal_management` (or any other aspect from the taxonomy)
- `--no-stream` to print the full answer at the end instead of streaming
- `--show-evidence` to print the prompt sent to Sonnet without firing the API

Backend (FastAPI with SSE):

```bash
uvicorn app.main:app --reload --port 8000
```

Frontend (separate terminal, from `frontend/`):

```bash
npm install
npm run dev
```

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
  app/
    main.py                   FastAPI backend, SSE streaming, session storage
  agent/
    graph.py                  LangGraph query understanding agent
    nodes.py                  parse, validate, retrieve, decline, generate nodes
  frontend/                   React + Vite client
  scripts/
    absa_batch_async.py       active ABSA pipeline
    build_counts.py           builds SQLite from aspects_long.csv
    ingest_chroma_bge.py      embeds reviews into ChromaDB
    retrieve.py               retrieval pipeline (filter, agg, rerank)
    answer.py                 LLM answer generation with quote validation
    test_agent.py             agent CLI smoke test
    test_cohere.py            Cohere rerank connection probe
    inspect_data.py           corpus structural summary
    run_eval.py               eval runner (4 passes, scored)
    legacy/                   superseded scripts kept for reference
  src/
    clean_reviews.py          cleaning pipeline
  requirements.txt
  README.md
```

## Costs

| Provider             | Spend             | Notes                               |
| -------------------- | ----------------- | ----------------------------------- |
| Outscraper           | ~$15              | One-time scrape                     |
| Anthropic ABSA batch | $3.13             | 3,802 reviews at 50% batch discount |
| Anthropic Sonnet     | ~$0.015 per query | Roughly $1-2 for the eval set       |
| Anthropic Haiku      | ~$0.001 per parse | Agent query understanding           |
| Cohere               | $0                | Free trial, 1,000 calls/month       |
| BGE-M3               | $0                | Local                               |

## Limits

- **Cleaned-query amplification on multi-turn merge.** When a user pivots scope mid-conversation with hedged language ("actually let's try X"), the merge prompt occasionally over-resets and drops aspects that should inherit. The system is robust on additive turns ("under £25" after an atmosphere query inherits cleanly).
- **Cross-lounge quote validation gap.** The validator confirms a quoted span exists in the verified pool, but doesn't confirm it's attributed to the lounge it's claimed for. A v2 fix would parse the claimed lounge from surrounding prose and validate against only that lounge's quote pool.
- **Geographic coverage.** The 14 lounges cover North, Central, East and South London. West London, Brixton specifically, and several Edgware Road venues are not in the dataset. Queries scoped to those areas correctly return no candidates and decline with a "no data" response rather than fabricating.
- **Single-user assumption.** Session state is held in-process. Concurrent users would share or trample state. Production would move sessions to Redis or similar.
- **BGE-M3 cold start.** First model load takes 20-40 seconds and ~2.3GB RAM. Once loaded, queries are fast.
- **Parser aggressiveness.** Aesthetic-loaded queries ("cheap and cheerful for a chilled evening") sometimes extract an aspect filter where the design intent was to leave it to semantic retrieval. Defensible but inconsistent.
- **Quote validation reads 0/0 when Sonnet paraphrases.** When the model writes an answer entirely in its own words without direct quotation, the validator has nothing to check. The metric is correct but ambiguous-looking; the system is functioning as designed.

## Responsible AI

- **Counts are deterministic, not generated.** Every count cited in every answer (e.g. "146 positive atmosphere reviews") is looked up from SQLite at query time. The LLM receives counts as pre-formatted strings and is forbidden from inventing numbers in the system prompt.
- **Quote validation on every answer.** Each double-quoted span in Sonnet's output is checked against the verified quote pool. Mismatches are flagged inline with `[unverified]` markers rather than silently rewritten. During testing, this caught hallucinated paraphrases that Sonnet had presented as direct quotes.
- **Honest acknowledgment of data gaps.** Queries scoped to geographies or topics outside the dataset return a "no data" response rather than fabricating. West London, Brixton, and halal-specific queries all behave this way.
- **Out-of-taxonomy gate.** Off-topic queries (e.g. "what's the weather") are detected at the parse stage and routed to a fast-decline path that skips retrieval and Sonnet entirely, returning a hardcoded scope message.
- **Recency weighting visible in evidence.** Each lounge's mean recency weight is shown in the UI alongside the answer. Older data is surfaced honestly rather than hidden.
- **"Limited evidence" labelling.** When a lounge's positive count for a queried aspect is below 5 reviews, the answer flags this inline ("limited evidence (4 reviews)"). The system calibrates its own confidence in-prose.
- **No PII collected.** Reviews are public Google Maps content. Reviewer names and profile photos are dropped at the cleaning stage; only review text, date, and rating are retained.
- **Logging.** FastAPI access logs to stdout. No per-user data is persisted beyond the in-process session store, which is cleared when the server restarts.

## Deployment

HookahLedger runs locally for the demo. The system's runtime footprint exceeds typical free-tier hosting: BGE-M3 holds ~2.3GB of model weights in memory once loaded, on top of the ChromaDB vector store and SQLite database on disk. Most free-tier hosts (Render free, Railway hobby, Fly.io shared-cpu-1x) cap at 256MB-512MB RAM, which is insufficient.

Production deployment would target a host with 4GB+ RAM at roughly $7-25/month: Render Standard, Railway, or Fly.io with a persistent volume. The frontend would deploy independently to Vercel or similar as a static bundle pointing at the backend's URL.

ChromaDB and SQLite are both single-file portable, so deployment is a matter of provisioning compute and copying the artefacts across rather than re-running the ingestion pipeline (which takes ~1 hour and incurs Anthropic ABSA costs). The Anthropic and Cohere API keys would move to the host's secrets manager.
