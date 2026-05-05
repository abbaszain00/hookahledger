# Legacy scripts

Superseded scripts kept for archival reference. Not part of the live pipeline.
These will not run without their original dependencies (e.g. `openai`,
`google-genai`) which have been dropped from `requirements.txt`.

- `absa_pilot.py`, `absa_pilot_v2.py`: early ABSA prompt iterations.
  Replaced by the validated v2 prompt embedded in `scripts/absa_batch_async.py`.
- `absa_batch.py`: synchronous ABSA with ThreadPoolExecutor concurrency.
  Replaced by `scripts/absa_batch_async.py`, which uses Anthropic's Message
  Batches API (50% cheaper).
- `ingest_chroma.py`: Gemini Embedding 001 ingestion. Replaced by
  `scripts/ingest_chroma_bge.py` after the Gemini free tier daily cap
  blocked corpus-scale embedding.
- `test_gemini.py`: one-shot Gemini API connection probe.
