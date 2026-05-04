import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const API_BASE = "http://localhost:8000";

const AREAS = [
  { value: "", label: "Any area" },
  { value: "North London", label: "North London" },
  { value: "Central London", label: "Central London" },
  { value: "East London", label: "East London" },
  { value: "South London", label: "South London" },
];

const PRICE_TIERS = [
  { value: "", label: "Any price" },
  { value: "budget", label: "Budget (~£15)" },
  { value: "mid", label: "Mid (~£20-25)" },
  { value: "premium", label: "Premium (£30+)" },
];

const ASPECTS = [
  { value: "", label: "No aspect filter" },
  { value: "flavour_quality", label: "Flavour quality" },
  { value: "coal_management", label: "Coal management" },
  { value: "service_speed", label: "Service speed" },
  { value: "value_for_money", label: "Value for money" },
  { value: "atmosphere_vibe", label: "Atmosphere" },
  { value: "seating_comfort", label: "Seating comfort" },
  { value: "food_quality", label: "Food quality" },
  { value: "wait_time", label: "Wait time" },
];

// Human labels for aspect codes when surfaced in the inferred-filters pill row.
const ASPECT_LABELS = Object.fromEntries(
  ASPECTS.filter((a) => a.value).map((a) => [a.value, a.label]),
);

// Human labels for price-tier codes.
const PRICE_LABELS = Object.fromEntries(
  PRICE_TIERS.filter((p) => p.value).map((p) => [p.value, p.label]),
);

// Vignette layered on the page background. Subtle radial falloff so the page
// reads as a low-lit space rather than a flat dark surface. Done as a CSS
// background image rather than an extra element so it covers the entire
// document height including the area below the content column.
const PAGE_VIGNETTE = {
  backgroundImage:
    "radial-gradient(ellipse 80% 60% at 50% 0%, rgba(232, 160, 76, 0.04) 0%, transparent 60%), radial-gradient(ellipse 100% 80% at 50% 100%, rgba(0, 0, 0, 0.4) 0%, transparent 70%)",
};

export default function App() {
  const [mode, setMode] = useState("filtered"); // "filtered" | "agent"

  const [query, setQuery] = useState("");
  const [area, setArea] = useState("");
  const [priceTier, setPriceTier] = useState("");
  const [aspectPositive, setAspectPositive] = useState("");

  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Agent-only state
  const [statusMessage, setStatusMessage] = useState(null); // current phase blurb
  const [parsed, setParsed] = useState(null); // parsed event payload

  // Hold the EventSource so we can close it on unmount or on a new query
  const eventSourceRef = useRef(null);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  function handleSubmit() {
    if (!query.trim() || streaming) return;

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    setStreaming(true);
    setStreamingText("");
    setError(null);
    setResult(null);
    setStatusMessage(null);
    setParsed(null);

    let url;
    if (mode === "agent") {
      const params = new URLSearchParams({ query: query.trim() });
      url = `${API_BASE}/api/agent/stream?${params.toString()}`;
    } else {
      const params = new URLSearchParams({ query: query.trim() });
      if (area) params.set("area", area);
      if (priceTier) params.set("price_tier", priceTier);
      if (aspectPositive) params.set("aspect_positive", aspectPositive);
      url = `${API_BASE}/api/chat/stream?${params.toString()}`;
    }

    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.addEventListener("status", (e) => {
      try {
        const data = JSON.parse(e.data);
        setStatusMessage(data.message);
      } catch (err) {
        console.error("Failed to parse status:", err, e.data);
      }
    });

    es.addEventListener("parsed", (e) => {
      try {
        const data = JSON.parse(e.data);
        setParsed(data);
      } catch (err) {
        console.error("Failed to parse parsed event:", err, e.data);
      }
    });

    es.addEventListener("token", (e) => {
      try {
        const chunk = JSON.parse(e.data);
        setStreamingText((prev) => prev + chunk);
        setStatusMessage(null);
      } catch (err) {
        console.error("Failed to parse token:", err, e.data);
      }
    });

    es.addEventListener("evidence", (e) => {
      try {
        const evidence = JSON.parse(e.data);
        setResult(evidence);
        if (evidence.agent) {
          setParsed((prev) => ({ ...(prev || {}), ...evidence.agent }));
        }
      } catch (err) {
        console.error("Failed to parse evidence:", err);
        setError("Failed to parse evidence event");
      }
    });

    es.addEventListener("done", () => {
      es.close();
      eventSourceRef.current = null;
      setStreaming(false);
      setStatusMessage(null);
    });

    es.addEventListener("error", (e) => {
      const data = e.data;
      if (data) {
        try {
          const parsed = JSON.parse(data);
          setError(parsed.error || "Server error during streaming");
        } catch {
          setError("Server error during streaming");
        }
      } else {
        if (eventSourceRef.current === es) {
          setError("Connection lost during streaming");
        }
      }
      es.close();
      eventSourceRef.current = null;
      setStreaming(false);
      setStatusMessage(null);
    });
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  function handleModeChange(next) {
    if (streaming) return;
    setMode(next);
    setResult(null);
    setStreamingText("");
    setError(null);
    setStatusMessage(null);
    setParsed(null);
  }

  const displayedText = result?.answer_validated ?? streamingText;
  const showAnswerCard =
    streaming || streamingText || result || statusMessage || parsed;

  const placeholder =
    mode === "agent"
      ? "e.g. somewhere with great atmosphere in north london under £25"
      : "e.g. best service in north london";

  return (
    <div className="min-h-screen" style={PAGE_VIGNETTE}>
      <div className="max-w-6xl mx-auto px-8 py-12">
        <header className="mb-12">
          <h1 className="font-display text-6xl font-semibold text-cream-100 leading-none">
            HookahLedger
          </h1>
          <p className="mt-3 text-cream-300 text-lg italic">
            London shisha lounge intelligence engine.
          </p>
          <div className="mt-6 h-px w-16 bg-saffron-400/60" />
        </header>

        <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] gap-10">
          {/* Sidebar - hidden in agent mode */}
          {mode === "filtered" && (
            <aside className="space-y-5">
              <FilterSelect
                label="Area"
                value={area}
                onChange={setArea}
                options={AREAS}
                disabled={streaming}
              />
              <FilterSelect
                label="Price tier"
                value={priceTier}
                onChange={setPriceTier}
                options={PRICE_TIERS}
                disabled={streaming}
              />
              <FilterSelect
                label="Aspect"
                value={aspectPositive}
                onChange={setAspectPositive}
                options={ASPECTS}
                disabled={streaming}
              />
            </aside>
          )}

          {/* Main column - spans full width when sidebar is hidden */}
          <main className={mode === "agent" ? "md:col-span-2" : ""}>
            <ModeToggle
              mode={mode}
              onChange={handleModeChange}
              disabled={streaming}
            />

            <div className="flex gap-2 mb-8">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={placeholder}
                className="flex-1 px-4 py-3 bg-base-800 border border-base-600 rounded text-cream-100 placeholder:text-cream-500 focus:outline-none focus:border-saffron-400 focus:ring-1 focus:ring-saffron-400/50 transition"
                disabled={streaming}
              />
              <button
                onClick={handleSubmit}
                disabled={streaming || !query.trim()}
                className="px-6 py-3 bg-saffron-400 text-base-900 font-medium rounded hover:bg-saffron-400/90 disabled:bg-base-700 disabled:text-cream-500 disabled:cursor-not-allowed transition"
              >
                {streaming ? "Thinking…" : "Ask"}
              </button>
            </div>

            {error && (
              <div className="p-4 mb-4 bg-base-800 border border-terracotta-500/40 rounded text-terracotta-500 text-sm">
                <span className="font-semibold">Error:</span> {error}
              </div>
            )}

            {showAnswerCard && (
              <div className="space-y-8">
                {/* Inferred filters - agent mode only, hidden on decline */}
                {mode === "agent" && parsed && !result?.is_declined && (
                  <InferredFilters parsed={parsed} />
                )}

                {/* Answer */}
                <div
                  className={`relative p-7 bg-base-800 rounded ${
                    result?.is_declined
                      ? "border border-base-600"
                      : "border-l-2 border-l-saffron-400 border-y border-r border-y-base-600 border-r-base-600"
                  }`}
                >
                  <h2 className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cream-300 mb-4 flex items-center gap-2">
                    {result?.is_declined ? "Outside system scope" : "Answer"}
                    {streaming && !result?.is_declined && (
                      <span className="inline-block w-1.5 h-1.5 rounded-full bg-saffron-400 animate-pulse shadow-[0_0_8px_rgba(232,160,76,0.6)]" />
                    )}
                  </h2>
                  {statusMessage && !result?.is_declined && (
                    <div className="text-xs text-cream-300 italic mb-4 -mt-2">
                      {statusMessage}
                    </div>
                  )}
                  <div className="prose prose-invert prose-sm max-w-none prose-headings:font-display prose-headings:text-cream-100 prose-headings:font-semibold prose-strong:text-cream-100 prose-p:text-cream-100 prose-p:leading-relaxed prose-li:text-cream-100 prose-em:text-cream-300">
                    {displayedText ? (
                      <ReactMarkdown>{displayedText}</ReactMarkdown>
                    ) : (
                      <span className="text-cream-300 italic">
                        {statusMessage || "Searching reviews…"}
                      </span>
                    )}
                  </div>
                </div>

                {/* Evidence cards (only after stream completes) */}
                {result?.lounges?.length > 0 && (
                  <div>
                    <h2 className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cream-300 mb-4">
                      Evidence ({result.lounges.length}{" "}
                      {result.lounges.length === 1 ? "lounge" : "lounges"})
                    </h2>
                    <div className="space-y-4">
                      {result.lounges.map((lounge) => (
                        <LoungeCard key={lounge.lounge_id} lounge={lounge} />
                      ))}
                    </div>
                  </div>
                )}

                {/* Metadata - suppressed on decline (zeros would be misleading) */}
                {result && !result.is_declined && (
                  <div className="text-xs text-cream-500 pt-5 border-t border-base-600 space-y-3">
                    <div className="flex flex-wrap gap-2">
                      {result.degraded && (
                        <div className="inline-block px-2.5 py-1 bg-base-800 border border-saffron-400/40 rounded text-saffron-400">
                          Partial response — stream interrupted before
                          completion.
                        </div>
                      )}
                      {result.rerank_succeeded === false && (
                        <div className="inline-block px-2.5 py-1 bg-base-800 border border-saffron-400/40 rounded text-saffron-400">
                          Rerank unavailable — results ordered by similarity and
                          recency only.
                        </div>
                      )}
                    </div>
                    <div className="font-mono tabular">
                      {result.candidates_pulled} candidates pulled ·{" "}
                      {result.chunks?.length || 0} chunks · {result.tokens_in}{" "}
                      in / {result.tokens_out} out · $
                      {result.cost_usd?.toFixed(4)} ·{" "}
                      {result.quote_validations?.filter((v) => v.valid)
                        .length || 0}
                      /{result.quote_validations?.length || 0} quotes verified
                    </div>
                  </div>
                )}
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}

function ModeToggle({ mode, onChange, disabled }) {
  return (
    <div className="mb-5 inline-flex p-0.5 bg-base-800 border border-base-600 rounded">
      <button
        onClick={() => onChange("filtered")}
        disabled={disabled}
        className={`px-4 py-1.5 text-xs uppercase tracking-widest font-medium rounded-sm transition ${
          mode === "filtered"
            ? "bg-saffron-400 text-base-900"
            : "text-cream-300 hover:text-cream-100"
        } disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        Filtered
      </button>
      <button
        onClick={() => onChange("agent")}
        disabled={disabled}
        className={`px-4 py-1.5 text-xs uppercase tracking-widest font-medium rounded-sm transition ${
          mode === "agent"
            ? "bg-saffron-400 text-base-900"
            : "text-cream-300 hover:text-cream-100"
        } disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        Agent
      </button>
    </div>
  );
}

function InferredFilters({ parsed }) {
  const pills = [];
  if (parsed.area)
    pills.push({ key: "area", label: "Area", value: parsed.area });
  if (parsed.price_tier)
    pills.push({
      key: "price",
      label: "Price",
      value: PRICE_LABELS[parsed.price_tier] || parsed.price_tier,
    });
  if (parsed.aspect_positive)
    pills.push({
      key: "aspect",
      label: "Aspect",
      value: ASPECT_LABELS[parsed.aspect_positive] || parsed.aspect_positive,
    });

  const parseFailed = parsed.parse_valid === false;
  const noFilters = pills.length === 0 && !parseFailed;

  return (
    <div className="p-4 bg-base-800 border border-base-600 rounded">
      <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cream-300 mb-3">
        Inferred from your query
      </div>
      {parseFailed ? (
        <div className="text-sm text-cream-300 italic">
          Could not infer structured filters
          {parsed.validation_reason
            ? ` (${parsed.validation_reason}).`
            : "."}{" "}
          Falling back to unfiltered retrieval.
        </div>
      ) : noFilters ? (
        <div className="text-sm text-cream-300 italic">
          No filters inferred. Searching all lounges.
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          {pills.map((p) => (
            <span
              key={p.key}
              className="inline-flex items-center gap-1.5 px-3 py-1 bg-base-700 border border-base-600 rounded-sm text-xs"
            >
              <span className="text-cream-300">{p.label}:</span>
              <span className="font-medium text-cream-100">{p.value}</span>
            </span>
          ))}
          {parsed.cleaned_query &&
            parsed.cleaned_query !== parsed.raw_query && (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-base-700 border border-base-600 rounded-sm text-xs">
                <span className="text-cream-300">Searching for:</span>
                <span className="font-medium text-cream-100 italic">
                  "{parsed.cleaned_query}"
                </span>
              </span>
            )}
        </div>
      )}
    </div>
  );
}

function FilterSelect({ label, value, onChange, options, disabled }) {
  return (
    <div>
      <label className="block text-[10px] font-semibold uppercase tracking-[0.2em] text-cream-300 mb-2">
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="w-full px-3 py-2 bg-base-800 border border-base-600 rounded text-sm text-cream-100 focus:outline-none focus:border-saffron-400 focus:ring-1 focus:ring-saffron-400/50 transition"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value} className="bg-base-800">
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

function LoungeCard({ lounge }) {
  const topAspects = (lounge.aspect_counts || [])
    .slice()
    .sort((a, b) => b.n_reviews - a.n_reviews)
    .slice(0, 5);

  const topChunk = lounge.chunks?.[0];
  const reviewExcerpt = topChunk ? extractReview(topChunk.document) : null;

  return (
    <div className="p-6 bg-base-800 border border-base-600 rounded">
      <div className="flex items-baseline justify-between mb-4 flex-wrap gap-2">
        <h3 className="font-display text-2xl font-semibold text-cream-100 leading-none">
          {lounge.lounge_name}
        </h3>
        <div className="text-xs text-cream-300 font-mono tabular">
          {lounge.area} · {lounge.total_reviews} reviews · recency{" "}
          {lounge.mean_recency_weight.toFixed(2)}
        </div>
      </div>

      {topAspects.length > 0 && (
        <div className="space-y-1.5 mb-4">
          {topAspects.map((a) => (
            <AspectRow key={`${a.aspect}-${a.sentiment}`} aspect={a} />
          ))}
        </div>
      )}

      {reviewExcerpt && (
        <div className="text-sm text-cream-100/90 italic border-l border-saffron-400/40 pl-4 mt-4 leading-relaxed">
          "{reviewExcerpt}"
          {topChunk?.review_date && (
            <span className="block not-italic text-xs text-cream-500 mt-2 font-mono tabular">
              — {topChunk.review_date}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function AspectRow({ aspect }) {
  const colour =
    aspect.sentiment === "positive"
      ? "text-sage-500"
      : aspect.sentiment === "negative"
        ? "text-terracotta-500"
        : "text-saffron-400";

  const label = aspect.aspect.replace(/_/g, " ");

  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-cream-100">{label}</span>
      <span className={`font-mono tabular text-xs ${colour}`}>
        {aspect.sentiment} · {aspect.n_reviews}
      </span>
    </div>
  );
}

function extractReview(document) {
  const idx = document.indexOf("Review: ");
  if (idx === -1) return null;
  const text = document.slice(idx + 8);
  return text.length > 300 ? text.slice(0, 300) + "…" : text;
}
