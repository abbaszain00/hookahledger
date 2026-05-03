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

    // Build the URL. Filtered path takes user-set filters as params;
    // agent path takes only the query.
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

    // Status events: agent path only. Inline phase line that updates in place.
    es.addEventListener("status", (e) => {
      try {
        const data = JSON.parse(e.data);
        setStatusMessage(data.message);
      } catch (err) {
        console.error("Failed to parse status:", err, e.data);
      }
    });

    // Parsed event: agent path only. Fires once the parse + validate nodes
    // have run. May contain mostly-null fields if parse_valid is false.
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
        // First token means generation has started; clear the status line so
        // the answer can take over the visual focus.
        setStatusMessage(null);
      } catch (err) {
        console.error("Failed to parse token:", err, e.data);
      }
    });

    es.addEventListener("evidence", (e) => {
      try {
        const evidence = JSON.parse(e.data);
        setResult(evidence);
        // If the agent block came in here as well, prefer it over the standalone
        // parsed event (same data, but the evidence block is canonical).
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
    // Clear results when switching modes so the user doesn't see filtered-mode
    // output sitting next to an agent-mode query box.
    setResult(null);
    setStreamingText("");
    setError(null);
    setStatusMessage(null);
    setParsed(null);
  }

  // Decide what text to display: validated text from the evidence event if
  // we have it, otherwise the in-progress streaming text.
  const displayedText = result?.answer_validated ?? streamingText;
  const showAnswerCard =
    streaming || streamingText || result || statusMessage || parsed;

  // Build the placeholder so it hints at the right kind of query for the mode.
  const placeholder =
    mode === "agent"
      ? "e.g. somewhere with great atmosphere in north london under £25"
      : "e.g. best service in north london";

  return (
    <div className="min-h-screen bg-stone-50">
      <div className="max-w-6xl mx-auto p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold">HookahLedger</h1>
          <p className="text-stone-600">
            London shisha lounge intelligence engine. Ask anything.
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] gap-8">
          {/* Sidebar - hidden in agent mode */}
          {mode === "filtered" && (
            <aside className="space-y-4">
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
            {/* Mode toggle */}
            <ModeToggle
              mode={mode}
              onChange={handleModeChange}
              disabled={streaming}
            />

            <div className="flex gap-2 mb-6">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={placeholder}
                className="flex-1 px-4 py-2 border border-stone-300 rounded focus:outline-none focus:border-stone-500"
                disabled={streaming}
              />
              <button
                onClick={handleSubmit}
                disabled={streaming || !query.trim()}
                className="px-6 py-2 bg-stone-800 text-white rounded disabled:opacity-50 hover:bg-stone-700"
              >
                {streaming ? "Thinking…" : "Ask"}
              </button>
            </div>

            {error && (
              <div className="p-4 mb-4 bg-red-50 border border-red-200 rounded text-red-900 text-sm">
                <strong>Error:</strong> {error}
              </div>
            )}

            {showAnswerCard && (
              <div className="space-y-6">
                {/* Inferred filters - agent mode only */}
                {mode === "agent" && parsed && (
                  <InferredFilters parsed={parsed} />
                )}

                {/* Answer */}
                <div className="p-6 bg-white border border-stone-200 rounded">
                  <h2 className="text-sm font-semibold uppercase tracking-wide text-stone-500 mb-3 flex items-center gap-2">
                    Answer
                    {streaming && (
                      <span className="inline-block w-2 h-2 rounded-full bg-stone-400 animate-pulse" />
                    )}
                    {statusMessage && (
                      <span className="text-stone-400 normal-case font-normal tracking-normal italic ml-2">
                        {statusMessage}
                      </span>
                    )}
                  </h2>
                  <div className="prose prose-stone prose-sm max-w-none">
                    {displayedText ? (
                      <ReactMarkdown>{displayedText}</ReactMarkdown>
                    ) : (
                      <span className="text-stone-400 italic">
                        {statusMessage || "Searching reviews…"}
                      </span>
                    )}
                  </div>
                </div>

                {/* Evidence cards (only after stream completes) */}
                {result?.lounges?.length > 0 && (
                  <div>
                    <h2 className="text-sm font-semibold uppercase tracking-wide text-stone-500 mb-3">
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

                {/* Metadata */}
                {result && (
                  <div className="text-xs text-stone-500 pt-4 border-t border-stone-200 space-y-2">
                    <div className="flex flex-wrap gap-2">
                      {result.degraded && (
                        <div className="inline-block px-2 py-1 bg-amber-50 border border-amber-200 rounded text-amber-900">
                          Partial response - stream interrupted before
                          completion.
                        </div>
                      )}
                      {result.rerank_succeeded === false && (
                        <div className="inline-block px-2 py-1 bg-amber-50 border border-amber-200 rounded text-amber-900">
                          Rerank unavailable - results ordered by similarity +
                          recency only.
                        </div>
                      )}
                    </div>
                    <div>
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
    <div className="mb-4 inline-flex p-1 bg-stone-200 rounded">
      <button
        onClick={() => onChange("filtered")}
        disabled={disabled}
        className={`px-4 py-1.5 text-sm font-medium rounded transition ${
          mode === "filtered"
            ? "bg-white text-stone-900 shadow-sm"
            : "text-stone-600 hover:text-stone-900"
        } disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        Filtered
      </button>
      <button
        onClick={() => onChange("agent")}
        disabled={disabled}
        className={`px-4 py-1.5 text-sm font-medium rounded transition ${
          mode === "agent"
            ? "bg-white text-stone-900 shadow-sm"
            : "text-stone-600 hover:text-stone-900"
        } disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        Agent
      </button>
    </div>
  );
}

function InferredFilters({ parsed }) {
  // Build pill list from whichever fields the parser populated.
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

  // parse_valid === false means the parser bailed (low confidence, schema fail,
  // API error). Show that rather than an empty pill row so the user understands
  // why no filters were applied.
  const parseFailed = parsed.parse_valid === false;
  const noFilters = pills.length === 0 && !parseFailed;

  return (
    <div className="p-4 bg-stone-100 border border-stone-200 rounded">
      <div className="text-xs font-semibold uppercase tracking-wide text-stone-500 mb-2">
        Inferred from your query
      </div>
      {parseFailed ? (
        <div className="text-sm text-stone-600 italic">
          Could not infer structured filters
          {parsed.validation_reason
            ? ` (${parsed.validation_reason}).`
            : "."}{" "}
          Falling back to unfiltered retrieval.
        </div>
      ) : noFilters ? (
        <div className="text-sm text-stone-600 italic">
          No filters inferred - searching all lounges.
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          {pills.map((p) => (
            <span
              key={p.key}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-white border border-stone-300 rounded text-xs"
            >
              <span className="text-stone-500">{p.label}:</span>
              <span className="font-medium text-stone-900">{p.value}</span>
            </span>
          ))}
          {parsed.cleaned_query &&
            parsed.cleaned_query !== parsed.raw_query && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-white border border-stone-300 rounded text-xs">
                <span className="text-stone-500">Searching for:</span>
                <span className="font-medium text-stone-900 italic">
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
      <label className="block text-xs font-semibold uppercase tracking-wide text-stone-500 mb-1">
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="w-full px-3 py-2 border border-stone-300 rounded bg-white text-sm focus:outline-none focus:border-stone-500"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
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
    <div className="p-5 bg-white border border-stone-200 rounded">
      <div className="flex items-baseline justify-between mb-3 flex-wrap gap-2">
        <h3 className="text-lg font-semibold">{lounge.lounge_name}</h3>
        <div className="text-xs text-stone-500">
          {lounge.area} · {lounge.total_reviews} reviews · recency{" "}
          {lounge.mean_recency_weight.toFixed(2)}
        </div>
      </div>

      {topAspects.length > 0 && (
        <div className="space-y-1 mb-3">
          {topAspects.map((a) => (
            <AspectRow key={`${a.aspect}-${a.sentiment}`} aspect={a} />
          ))}
        </div>
      )}

      {reviewExcerpt && (
        <div className="text-sm text-stone-700 italic border-l-2 border-stone-300 pl-3 mt-3">
          "{reviewExcerpt}"
          {topChunk?.review_date && (
            <span className="block not-italic text-xs text-stone-500 mt-1">
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
      ? "text-emerald-700"
      : aspect.sentiment === "negative"
        ? "text-red-700"
        : "text-amber-700";

  const label = aspect.aspect.replace(/_/g, " ");

  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-stone-700">{label}</span>
      <span className={`font-mono ${colour}`}>
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
