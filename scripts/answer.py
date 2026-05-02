"""
answer.py - LLM answer generation on top of retrieval.

End-to-end:
    query  ->  retrieve.RetrievalPipeline  ->  formatted evidence prompt
           ->  Anthropic Sonnet  ->  cited answer streamed to stdout

What this script enforces (design doc requirements):
  - Counts come from SQLite, NEVER from the model. We hand them to Sonnet
    pre-formatted as plain numbers; the system prompt forbids invention.
  - Quotes are taken verbatim from retrieved chunks. The system prompt
    requires the model to use them as-is.
  - Thin-evidence flagging: the prompt makes Sonnet say "I don't have
    enough data on this" when retrieved evidence is below threshold.
  - Negative-evidence handling: the prompt requires Sonnet to recommend
    against a lounge if the evidence is predominantly negative.

Usage:
  CLI:
    python scripts/answer.py "best coal management in north london"
    python scripts/answer.py "good for date night" --area "Central London"
    python scripts/answer.py "best value under 25" --no-stream

  As module:
    from scripts.answer import AnswerEngine
    engine = AnswerEngine()
    result = engine.answer("best mint flavour")
    print(result.text)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import anthropic
from dotenv import load_dotenv

# Reuse the retrieval pipeline from the same scripts/ folder
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import (  # noqa: E402  (after sys.path tweak)
    LoungeEvidence,
    RetrievalPipeline,
    RetrievalResult,
    RetrievedChunk,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 1500
THIN_EVIDENCE_THRESHOLD = 5  # less than 5 reviews on a topic -> flag


# ---------------------------------------------------------------------------
# System prompt (per Day 1 design doc, lightly tightened)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are HookahLedger, a London shisha lounge expert. Answer ONLY using the evidence provided below the user's question. Do not speculate or rely on outside knowledge.

<rules>
1. Every claim must cite a specific aspect and the review count, e.g. "based on 14 reviews mentioning coal management".
2. Review counts are provided as structured data in <evidence_counts>. Use them EXACTLY as given. Never invent, round, or adjust them.
3. ABSOLUTE QUOTE RULE: Every double-quoted phrase in your answer MUST be copied character-for-character from <verified_quotes>. If a quote you want to use does not appear in <verified_quotes>, you cannot use it - rewrite the sentence in your own words without quotation marks. Quotes from <evidence_chunks> have NOT been verified and may not be quoted directly; use that section only for context, not for quotations.
4. Do not paraphrase, embellish, edit, combine, or invent quotes. Do not concatenate two real quotes into one. Do not add filler words to a real quote.
5. Choose quotes that directly support the claim you are making. If <verified_quotes> contains nothing that supports a particular claim, describe the evidence in your own words without using quotation marks.
6. If the count for a topic is below {THIN_EVIDENCE_THRESHOLD}, explicitly flag it: "limited evidence ({{count}} reviews)".
7. If no evidence exists for the user's question, say so directly: "I don't have data on this." Do not pad with unrelated content.
8. If retrieved evidence is predominantly negative for a lounge, your recommendation must reflect that. Do not soften or hedge negative findings.
</rules>

<response_format>
Use British English. Write in a calm, factual register - not breathless marketing copy. The reader is an adult who wants accurate information, not hype.

Structure:
1. Direct recommendation (1-2 sentences).
2. Evidence breakdown - bullet points: aspect, sentiment, count, key quote (only when a real verified quote supports it).
3. Caveats - data gaps, low-evidence flags, recency notes.

Do not use phrases like "I hope this helps" or "let me know if you have questions". Just deliver the answer.
</response_format>

<self_check>
Before finalising your answer, scan every double-quoted span in your output. For each one, confirm the exact characters appear inside the <verified_quotes> block. If any quote fails this check, rewrite that sentence without quotation marks. The system will validate quotes after you respond - quotes that don't match will be flagged.
</self_check>"""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class QuoteValidation:
    quote: str
    valid: bool


@dataclass
class AnswerResult:
    query: str
    text: str                       # the LLM's raw answer (may contain hallucinated quotes)
    text_validated: str              # answer with hallucinated quotes neutralised
    retrieval: RetrievalResult
    verified_quotes: dict           # {lounge_id: [quote_row, ...]}
    quote_validations: list[QuoteValidation]  # one per quote in the answer
    tokens_in: int
    tokens_out: int
    cost_usd: float


# ---------------------------------------------------------------------------
# Quote handling: normalisation + extraction + validation
# ---------------------------------------------------------------------------
# Match double-quoted spans. Keeps it simple - assumes quotes don't span
# multiple paragraphs (true in practice for our outputs).
QUOTE_PATTERN = re.compile(r'"([^"\n]+?)"')


def normalise_for_compare(s: str) -> str:
    """Normalise a string for substring comparison.

    Same conventions as absa_batch (NFKC + smart quotes + dashes -> ASCII).
    Matters for quote validation because Sonnet may emit smart quotes even
    when the source uses straight ones.
    """
    s = unicodedata.normalize("NFKC", s)
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-",
        "\u2026": "...",
        "\u00A0": " ",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s


def extract_quotes(text: str) -> list[str]:
    """Pull every double-quoted span out of the answer.

    Normalises the text first so smart quotes are mapped to straight quotes
    before the regex runs - otherwise a Sonnet response using curly quotes
    would slip past extraction entirely and never reach validation.
    """
    return QUOTE_PATTERN.findall(normalise_for_compare(text))


def validate_quote(quote: str, verified_quotes: dict) -> bool:
    """Return True if `quote` is a substring of at least one verified quote
    across all lounges. Comparison is Unicode-normalised."""
    qn = normalise_for_compare(quote.strip())
    if not qn:
        return False
    for lounge_quotes in verified_quotes.values():
        for entry in lounge_quotes:
            ref = normalise_for_compare(entry["quote"])
            if qn in ref:
                return True
    return False


def neutralise_invalid_quotes(text: str, verified_quotes: dict) -> tuple[str, list[QuoteValidation]]:
    """Find quoted spans in `text`. Each one is checked against verified_quotes.

    Returns (neutralised_text, validations).
    For invalid quotes, drop the quote marks and append [unverified] so the
    reader knows the system flagged it. We keep the inline content rather
    than redacting outright because (a) it's still useful context and (b)
    deleting text mid-paragraph leaves grammatical wreckage.

    The input text is normalised first (smart quotes -> straight quotes etc).
    Without normalisation, Sonnet responses using curly quotes would never
    match the regex and would skip validation silently.
    """
    normalised = normalise_for_compare(text)
    validations: list[QuoteValidation] = []

    def _replace(match: re.Match) -> str:
        original = match.group(1)
        valid = validate_quote(original, verified_quotes)
        validations.append(QuoteValidation(quote=original, valid=valid))
        if valid:
            return f'"{original}"'
        return f"{original} [unverified]"

    new_text = QUOTE_PATTERN.sub(_replace, normalised)
    return new_text, validations


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------
def format_evidence(
    retrieval: RetrievalResult,
    verified_quotes: dict,
) -> str:
    """Turn a RetrievalResult into the structured prompt block we hand to Sonnet.

    Three sections:
      <evidence_counts>   - deterministic SQLite counts per (lounge, aspect, sentiment)
      <verified_quotes>   - pre-validated quotes, also from SQLite. THE ONLY SOURCE
                            the LLM is allowed to quote from.
      <evidence_chunks>   - top-K reranked review chunks, for context only.

    The model is instructed to use counts from the first section as-is and
    to quote ONLY from the second section. The third gives narrative context
    but is explicitly off-limits for direct quotation.
    """
    if not retrieval.lounges:
        return (
            "<evidence_counts>\n(no evidence retrieved)\n</evidence_counts>\n\n"
            "<verified_quotes>\n(no verified quotes)\n</verified_quotes>\n\n"
            "<evidence_chunks>\n(no chunks retrieved)\n</evidence_chunks>"
        )

    # ---- counts ----
    counts_lines: list[str] = []
    for lg in retrieval.lounges:
        counts_lines.append(
            f"\n[{lg.lounge_name}] (lounge_id={lg.lounge_id}, area={lg.area})"
        )
        counts_lines.append(
            f"  Total reviews scraped: {lg.total_reviews} | "
            f"aspect mentions extracted: {lg.total_aspect_mentions} | "
            f"mean recency weight: {lg.mean_recency_weight:.2f}"
        )
        if lg.aspect_counts:
            sorted_counts = sorted(
                lg.aspect_counts, key=lambda r: r["n_reviews"], reverse=True
            )
            for r in sorted_counts:
                thin_marker = " [LIMITED]" if r["n_reviews"] < THIN_EVIDENCE_THRESHOLD else ""
                counts_lines.append(
                    f"    - {r['aspect']:20s} {r['sentiment']:9s} "
                    f"= {r['n_reviews']} reviews{thin_marker}"
                )
        else:
            counts_lines.append("    (no aspect mentions in SQLite for this lounge)")

    # ---- verified quotes (from SQLite aspect_quotes) ----
    quote_lines: list[str] = []
    for lg in retrieval.lounges:
        lounge_quotes = verified_quotes.get(lg.lounge_id, [])
        if not lounge_quotes:
            continue
        quote_lines.append(
            f"\n[{lg.lounge_name}] (lounge_id={lg.lounge_id})"
        )
        for q in lounge_quotes:
            quote_lines.append(
                f"  - {q['aspect']} ({q['sentiment']}, {q['review_date']}): "
                f"\"{q['quote']}\""
            )
    if not quote_lines:
        quote_lines = ["\n(no verified quotes available for these lounges)"]

    # ---- chunks (context only) ----
    chunks_lines: list[str] = []
    for i, c in enumerate(retrieval.chunks, 1):
        review_start = c.document.find("Review: ")
        review_text = (
            c.document[review_start + 8:] if review_start > -1 else c.document
        )
        chunks_lines.append(
            f"\n[chunk {i}] lounge={c.lounge_id} | date={c.review_date} | "
            f"recency={c.recency_weight:.2f}"
            + (f" | rerank_relevance={c.cohere_relevance:.3f}"
               if c.cohere_relevance is not None else "")
        )
        chunks_lines.append(f"  aspects: {c.aspect_sentiments_csv or '(none)'}")
        chunks_lines.append(f"  review: {review_text}")

    return (
        "<evidence_counts>"
        + "\n".join(counts_lines)
        + "\n</evidence_counts>\n\n<verified_quotes>"
        + "\n".join(quote_lines)
        + "\n</verified_quotes>\n\n<evidence_chunks>"
        + "\n".join(chunks_lines)
        + "\n</evidence_chunks>"
    )


# ---------------------------------------------------------------------------
# Answer engine
# ---------------------------------------------------------------------------
class AnswerEngine:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        retrieval_pipeline: RetrievalPipeline | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parent.parent
        load_dotenv(project_root / ".env")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not in .env")

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
        self.retrieval = retrieval_pipeline or RetrievalPipeline()

    def _build_user_message(
        self,
        query: str,
        retrieval: RetrievalResult,
        verified_quotes: dict,
    ) -> str:
        filters_line = ""
        if any(retrieval.filters.values()):
            filters_line = (
                "\nFilters applied: "
                + ", ".join(f"{k}={v}" for k, v in retrieval.filters.items() if v)
                + "\n"
            )

        return (
            f"Question: {query}{filters_line}\n\n"
            + format_evidence(retrieval, verified_quotes)
        )

    def answer(
        self,
        query: str,
        area: str | None = None,
        price_tier: str | None = None,
        aspect_positive: str | None = None,
        top_k: int = 5,
        stream: bool = True,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> AnswerResult:
        retrieval = self.retrieval.retrieve(
            query=query,
            area=area,
            price_tier=price_tier,
            aspect_positive=aspect_positive,
            top_k=top_k,
        )

        # Pull verified quotes from SQLite for the lounges that survived retrieval
        unique_lounge_ids = list({lg.lounge_id for lg in retrieval.lounges})
        verified_quotes = self.retrieval.fetch_verified_quotes(unique_lounge_ids)

        user_message = self._build_user_message(query, retrieval, verified_quotes)

        if stream:
            text, tokens_in, tokens_out = self._stream(user_message, max_tokens)
        else:
            text, tokens_in, tokens_out = self._oneshot(user_message, max_tokens)

        # Post-generation: scan every quoted span and check it's a substring
        # of one of the verified quotes. Invalid ones get the [unverified]
        # marker so the reader can see what the system flagged.
        text_validated, validations = neutralise_invalid_quotes(text, verified_quotes)

        # Sonnet 4.5 pricing: $3/M input, $15/M output
        cost_usd = (tokens_in / 1_000_000) * 3.0 + (tokens_out / 1_000_000) * 15.0

        return AnswerResult(
            query=query,
            text=text,
            text_validated=text_validated,
            retrieval=retrieval,
            verified_quotes=verified_quotes,
            quote_validations=validations,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )

    def _stream(self, user_message: str, max_tokens: int) -> tuple[str, int, int]:
        """Stream the answer to stdout as it arrives, return the full text and usage."""
        chunks: list[str] = []
        tokens_in = 0
        tokens_out = 0
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                chunks.append(text)
            print()  # newline after stream
            final = stream.get_final_message()
            if final.usage:
                tokens_in = final.usage.input_tokens
                tokens_out = final.usage.output_tokens
        return "".join(chunks), tokens_in, tokens_out

    def _oneshot(self, user_message: str, max_tokens: int) -> tuple[str, int, int]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        tokens_in = resp.usage.input_tokens if resp.usage else 0
        tokens_out = resp.usage.output_tokens if resp.usage else 0
        return text, tokens_in, tokens_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--area", default=None)
    parser.add_argument("--price-tier", default=None,
                        choices=["budget", "mid", "premium"])
    parser.add_argument("--aspect-positive", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-stream", action="store_true",
                        help="Wait for the full response instead of streaming")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--show-evidence", action="store_true",
                        help="Print the evidence block sent to Sonnet (debugging)")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON output (implies --no-stream)")
    args = parser.parse_args()

    engine = AnswerEngine()

    if args.show_evidence:
        # Run retrieval only and print the prompt that would be sent
        retrieval = engine.retrieval.retrieve(
            query=args.query,
            area=args.area,
            price_tier=args.price_tier,
            aspect_positive=args.aspect_positive,
            top_k=args.top_k,
        )
        unique_lounges = list({lg.lounge_id for lg in retrieval.lounges})
        verified_quotes = engine.retrieval.fetch_verified_quotes(unique_lounges)
        print(engine._build_user_message(args.query, retrieval, verified_quotes))
        return

    if args.json:
        result = engine.answer(
            query=args.query,
            area=args.area,
            price_tier=args.price_tier,
            aspect_positive=args.aspect_positive,
            top_k=args.top_k,
            stream=False,
            max_tokens=args.max_tokens,
        )
        out = {
            "query": result.query,
            "answer_raw": result.text,
            "answer_validated": result.text_validated,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "cost_usd": result.cost_usd,
            "quote_validations": [asdict(v) for v in result.quote_validations],
            "lounges_cited": [
                {
                    "lounge_id": lg.lounge_id,
                    "lounge_name": lg.lounge_name,
                    "total_reviews": lg.total_reviews,
                }
                for lg in result.retrieval.lounges
            ],
        }
        print(json.dumps(out, indent=2))
        return

    # Default: stream and print
    print(f"Query: {args.query}")
    if any([args.area, args.price_tier, args.aspect_positive]):
        filters = ", ".join(
            f"{k}={v}" for k, v in {
                "area": args.area,
                "price_tier": args.price_tier,
                "aspect_positive": args.aspect_positive,
            }.items() if v
        )
        print(f"Filters: {filters}")
    print("-" * 60)

    result = engine.answer(
        query=args.query,
        area=args.area,
        price_tier=args.price_tier,
        aspect_positive=args.aspect_positive,
        top_k=args.top_k,
        stream=not args.no_stream,
        max_tokens=args.max_tokens,
    )

    if args.no_stream:
        print(result.text)

    # Quote validation summary
    n_quotes = len(result.quote_validations)
    n_valid = sum(1 for v in result.quote_validations if v.valid)
    n_invalid = n_quotes - n_valid

    print("-" * 60)
    print(
        f"[Quotes: {n_valid}/{n_quotes} verified"
        + (f", {n_invalid} flagged" if n_invalid else "")
        + "]"
    )
    if n_invalid:
        print("[Invalid quotes (not in verified_quotes):]")
        for v in result.quote_validations:
            if not v.valid:
                print(f"  - \"{v.quote}\"")
        print("\n[Validated answer with [unverified] markers:]\n")
        print(result.text_validated)

    print("-" * 60)
    print(
        f"[Sonnet usage: {result.tokens_in} in / {result.tokens_out} out "
        f"= ${result.cost_usd:.4f}]"
    )
    print(
        f"[Retrieval: {len(result.retrieval.chunks)} chunks across "
        f"{len(result.retrieval.lounges)} lounges, "
        f"candidate pool {result.retrieval.candidates_pulled}]"
    )


if __name__ == "__main__":
    main()