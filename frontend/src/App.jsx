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

export default function App() {
  const [query, setQuery] = useState("");
  const [area, setArea] = useState("");
  const [priceTier, setPriceTier] = useState("");
  const [aspectPositive, setAspectPositive] = useState("");

  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Hold the EventSource so we can close it on unmount or on a new query
  const eventSourceRef = useRef(null);

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  function handleSubmit() {
    if (!query.trim() || streaming) return;

    // Tear down any prior stream
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    setStreaming(true);
    setStreamingText("");
    setError(null);
    setResult(null);

    // Build the URL with query params (EventSource only supports GET)
    const params = new URLSearchParams({ query: query.trim() });
    if (area) params.set("area", area);
    if (priceTier) params.set("price_tier", priceTier);
    if (aspectPositive) params.set("aspect_positive", aspectPositive);

    const url = `${API_BASE}/api/chat/stream?${params.toString()}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.addEventListener("token", (e) => {
      try {
        const chunk = JSON.parse(e.data);
        setStreamingText((prev) => prev + chunk);
      } catch (err) {
        console.error("Failed to parse token:", err, e.data);
      }
    });

    es.addEventListener("evidence", (e) => {
      try {
        const evidence = JSON.parse(e.data);
        setResult(evidence);
      } catch (err) {
        console.error("Failed to parse evidence:", err);
        setError("Failed to parse evidence event");
      }
    });

    es.addEventListener("done", () => {
      es.close();
      eventSourceRef.current = null;
      setStreaming(false);
    });

    es.addEventListener("error", (e) => {
      // EventSource fires 'error' for both server-sent error events AND
      // network/connection failures. The server-sent ones have a data field;
      // network errors don't.
      const data = e.data;
      if (data) {
        try {
          const parsed = JSON.parse(data);
          setError(parsed.error || "Server error during streaming");
        } catch {
          setError("Server error during streaming");
        }
      } else {
        // Connection error - only show if we weren't already done
        // (EventSource fires error after a clean close too)
        if (eventSourceRef.current === es) {
          setError("Connection lost during streaming");
        }
      }
      es.close();
      eventSourceRef.current = null;
      setStreaming(false);
    });
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  // Decide what text to display: validated text from the evidence event if
  // we have it, otherwise the in-progress streaming text.
  const displayedText = result?.answer_validated ?? streamingText;
  const showAnswerCard = streaming || streamingText || result;

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
          {/* Sidebar */}
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

          {/* Main column */}
          <main>
            <div className="flex gap-2 mb-6">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. best service in north london"
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
                {/* Answer */}
                <div className="p-6 bg-white border border-stone-200 rounded">
                  <h2 className="text-sm font-semibold uppercase tracking-wide text-stone-500 mb-3 flex items-center gap-2">
                    Answer
                    {streaming && (
                      <span className="inline-block w-2 h-2 rounded-full bg-stone-400 animate-pulse" />
                    )}
                  </h2>
                  <div className="prose prose-stone prose-sm max-w-none">
                    {displayedText ? (
                      <ReactMarkdown>{displayedText}</ReactMarkdown>
                    ) : (
                      <span className="text-stone-400 italic">
                        Searching reviews…
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
                  <div className="text-xs text-stone-500 pt-4 border-t border-stone-200">
                    {result.candidates_pulled} candidates pulled ·{" "}
                    {result.chunks?.length || 0} chunks · {result.tokens_in} in
                    / {result.tokens_out} out · ${result.cost_usd?.toFixed(4)} ·{" "}
                    {result.quote_validations?.filter((v) => v.valid).length ||
                      0}
                    /{result.quote_validations?.length || 0} quotes verified
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
