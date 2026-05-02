import { useState } from "react";
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

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  async function handleSubmit() {
    if (!query.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const body = { query: query.trim() };
      if (area) body.area = area;
      if (priceTier) body.price_tier = priceTier;
      if (aspectPositive) body.aspect_positive = aspectPositive;

      const response = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`API ${response.status}: ${text}`);
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

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
              disabled={loading}
            />
            <FilterSelect
              label="Price tier"
              value={priceTier}
              onChange={setPriceTier}
              options={PRICE_TIERS}
              disabled={loading}
            />
            <FilterSelect
              label="Aspect"
              value={aspectPositive}
              onChange={setAspectPositive}
              options={ASPECTS}
              disabled={loading}
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
                disabled={loading}
              />
              <button
                onClick={handleSubmit}
                disabled={loading || !query.trim()}
                className="px-6 py-2 bg-stone-800 text-white rounded disabled:opacity-50 hover:bg-stone-700"
              >
                {loading ? "Thinking…" : "Ask"}
              </button>
            </div>

            {error && (
              <div className="p-4 mb-4 bg-red-50 border border-red-200 rounded text-red-900 text-sm">
                <strong>Error:</strong> {error}
              </div>
            )}

            {loading && (
              <div className="p-4 mb-4 bg-stone-100 border border-stone-200 rounded text-stone-600 text-sm">
                Searching reviews and generating answer…
              </div>
            )}

            {result && <AnswerView result={result} />}
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

function AnswerView({ result }) {
  return (
    <div className="space-y-6">
      {/* Answer */}
      <div className="p-6 bg-white border border-stone-200 rounded">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-stone-500 mb-3">
          Answer
        </h2>
        <div className="prose prose-stone prose-sm max-w-none">
          <ReactMarkdown>{result.answer_validated}</ReactMarkdown>
        </div>
      </div>

      {/* Evidence cards */}
      {result.lounges?.length > 0 && (
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
      <div className="text-xs text-stone-500 pt-4 border-t border-stone-200">
        {result.candidates_pulled} candidates pulled ·{" "}
        {result.chunks?.length || 0} chunks · {result.tokens_in} in /{" "}
        {result.tokens_out} out · ${result.cost_usd?.toFixed(4)} ·{" "}
        {result.quote_validations?.filter((v) => v.valid).length || 0}/
        {result.quote_validations?.length || 0} quotes verified
      </div>
    </div>
  );
}

function LoungeCard({ lounge }) {
  // Top 5 aspect_counts by n_reviews, sorted descending
  const topAspects = (lounge.aspect_counts || [])
    .slice()
    .sort((a, b) => b.n_reviews - a.n_reviews)
    .slice(0, 5);

  // Top quote from the highest-scoring chunk for this lounge
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
  // The Chroma documents are formatted: "Lounge: ... | ... | Review: <text>"
  const idx = document.indexOf("Review: ");
  if (idx === -1) return null;
  const text = document.slice(idx + 8);
  // Trim to a reasonable length so the card doesn't blow up
  return text.length > 300 ? text.slice(0, 300) + "…" : text;
}
