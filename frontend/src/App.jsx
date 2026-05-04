import { useState, useRef, useEffect, useCallback } from "react";
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

const ASPECT_LABELS = Object.fromEntries(
  ASPECTS.filter((a) => a.value).map((a) => [a.value, a.label]),
);

const PRICE_LABELS = Object.fromEntries(
  PRICE_TIERS.filter((p) => p.value).map((p) => [p.value, p.label]),
);

// Human labels for lounge IDs. Keep in sync with data/lounges.csv.
// Used by InferredFilters to render lounge_focus as a name rather than the
// underlying ID. Missing keys fall back to the raw lounge_id.
const LOUNGE_LABELS = {
  noya_harringay: "Noya Shisha Lounge & Restaurant",
  shisha_garden_edgware: "The Shisha Garden",
  the_banc_seven_sisters: "The Banc",
  tigerbay_kingsbury: "TigerBay Shisha Lounge",
  shishawi_edgware: "Shishawi",
  aldar_edgware: "Al-Dar I",
  mamounia_edgware: "Mamounia Lounge",
  basrah_edgware: "Basrah Lounge",
  laika_soho: "Laika Soho",
  globe_lounge_forest_gate: "Globe Lounge",
  cafe_cairo_brixton: "Cafe Cairo",
  ground5_brixton: "Ground5 Shisha Lounge",
};

const PAGE_VIGNETTE = {
  backgroundImage:
    "radial-gradient(ellipse 80% 60% at 50% 0%, rgba(232, 160, 76, 0.04) 0%, transparent 60%), radial-gradient(ellipse 100% 80% at 50% 100%, rgba(0, 0, 0, 0.4) 0%, transparent 70%)",
};

function newSessionId() {
  // crypto.randomUUID is in all modern browsers; fallback if absent.
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function newTurn(query, mode) {
  return {
    id: `turn-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    query,
    mode,
    parsed: null,
    statusMessage: null,
    streamingText: "",
    result: null,
    error: null,
    streaming: true,
  };
}

export default function App() {
  const [mode, setMode] = useState("filtered");

  const [query, setQuery] = useState("");
  const [area, setArea] = useState("");
  const [priceTier, setPriceTier] = useState("");
  const [aspectPositive, setAspectPositive] = useState("");

  // The conversation: list of turns. Most recent turn is last.
  const [turns, setTurns] = useState([]);

  // Session ID survives across turns; rotates on "New conversation" or mode switch.
  const [sessionId, setSessionId] = useState(newSessionId);

  // Ephemeral notice shown when the conversation auto-resets.
  const [resetNotice, setResetNotice] = useState(null);

  // Refs
  const eventSourceRef = useRef(null);
  const inputRef = useRef(null);
  const scrollAnchorRef = useRef(null);

  // Whether anything is currently streaming. Derived from turns.
  const streaming = turns.length > 0 && turns[turns.length - 1].streaming;

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Scroll to bottom whenever a new turn arrives or tokens stream in.
  useEffect(() => {
    if (scrollAnchorRef.current) {
      scrollAnchorRef.current.scrollIntoView({
        behavior: "smooth",
        block: "end",
      });
    }
  }, [turns]);

  // Update the most recent turn (the in-progress one) with a partial update.
  // Wrapped in useCallback because the SSE handlers close over it.
  const updateLastTurn = useCallback((updater) => {
    setTurns((prev) => {
      if (prev.length === 0) return prev;
      const next = prev.slice();
      const last = next[next.length - 1];
      next[next.length - 1] =
        typeof updater === "function" ? updater(last) : { ...last, ...updater };
      return next;
    });
  }, []);

  function resetConversation(reason) {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    // Tell the backend to drop the session; fire-and-forget.
    if (sessionId) {
      fetch(`${API_BASE}/api/session/reset?session_id=${sessionId}`, {
        method: "POST",
      }).catch(() => {});
    }
    setSessionId(newSessionId());
    setTurns([]);
    setQuery("");
    if (reason) {
      setResetNotice(reason);
      setTimeout(() => setResetNotice(null), 3000);
    }
    if (inputRef.current) inputRef.current.focus();
  }

  function handleModeChange(next) {
    if (streaming) return;
    if (next === mode) return;
    setMode(next);
    // Switching modes mid-conversation breaks the conversational arc.
    // Reset so the chat log doesn't mix retrieval semantics.
    if (turns.length > 0) {
      resetConversation(`Switched to ${next} mode. Starting new conversation.`);
    }
  }

  function handleNewConversation() {
    if (streaming) return;
    if (turns.length === 0) return;
    resetConversation(null);
  }

  function handleSubmit() {
    if (!query.trim() || streaming) return;

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    const submittedQuery = query.trim();
    const submittedMode = mode;

    // Append the new turn first (in-progress).
    setTurns((prev) => [...prev, newTurn(submittedQuery, submittedMode)]);

    // Clear the input so the user can type the next question.
    setQuery("");

    // Build URL.
    const params = new URLSearchParams({ query: submittedQuery });
    if (sessionId) params.set("session_id", sessionId);
    let url;
    if (submittedMode === "agent") {
      url = `${API_BASE}/api/agent/stream?${params.toString()}`;
    } else {
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
        updateLastTurn({ statusMessage: data.message });
      } catch (err) {
        console.error("Failed to parse status:", err, e.data);
      }
    });

    es.addEventListener("parsed", (e) => {
      try {
        const data = JSON.parse(e.data);
        updateLastTurn({ parsed: data });
      } catch (err) {
        console.error("Failed to parse parsed event:", err, e.data);
      }
    });

    es.addEventListener("token", (e) => {
      try {
        const chunk = JSON.parse(e.data);
        updateLastTurn((t) => ({
          ...t,
          streamingText: t.streamingText + chunk,
          statusMessage: null, // tokens flowing means generation started
        }));
      } catch (err) {
        console.error("Failed to parse token:", err, e.data);
      }
    });

    es.addEventListener("evidence", (e) => {
      try {
        const evidence = JSON.parse(e.data);
        updateLastTurn((t) => ({
          ...t,
          result: evidence,
          parsed: evidence.agent
            ? { ...(t.parsed || {}), ...evidence.agent }
            : t.parsed,
        }));
      } catch (err) {
        console.error("Failed to parse evidence:", err);
        updateLastTurn({ error: "Failed to parse evidence event" });
      }
    });

    es.addEventListener("done", () => {
      es.close();
      eventSourceRef.current = null;
      updateLastTurn({ streaming: false, statusMessage: null });
      if (inputRef.current) inputRef.current.focus();
    });

    es.addEventListener("error", (e) => {
      const data = e.data;
      let message = null;
      if (data) {
        try {
          const parsed = JSON.parse(data);
          message = parsed.error || "Server error during streaming";
        } catch {
          message = "Server error during streaming";
        }
      } else {
        if (eventSourceRef.current === es) {
          message = "Connection lost during streaming";
        }
      }
      es.close();
      eventSourceRef.current = null;
      updateLastTurn({
        error: message,
        streaming: false,
        statusMessage: null,
      });
    });
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  const placeholder =
    mode === "agent"
      ? turns.length === 0
        ? "e.g. somewhere with great atmosphere in north london under £25"
        : "ask a follow-up..."
      : turns.length === 0
        ? "e.g. best service in north london"
        : "ask a follow-up...";

  return (
    <div className="min-h-screen flex flex-col" style={PAGE_VIGNETTE}>
      <div className="flex-1 max-w-6xl w-full mx-auto px-8 py-12 pb-44">
        <header className="mb-12 flex items-end justify-between flex-wrap gap-4">
          <div>
            <h1 className="font-display text-6xl font-semibold text-cream-100 leading-none">
              HookahLedger
            </h1>
            <p className="mt-3 text-cream-300 text-lg italic">
              London shisha lounge intelligence engine.
            </p>
            <div className="mt-6 h-px w-16 bg-saffron-400/60" />
          </div>
          {turns.length > 0 && (
            <button
              onClick={handleNewConversation}
              disabled={streaming}
              className="text-[10px] uppercase tracking-[0.2em] text-cream-300 hover:text-saffron-400 disabled:opacity-50 disabled:cursor-not-allowed transition pb-1 border-b border-transparent hover:border-saffron-400/40"
            >
              New conversation
            </button>
          )}
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
              {turns.length > 0 && (
                <div className="text-[10px] text-cream-500 italic leading-relaxed pt-2">
                  Filter changes apply to your next question, not previous ones.
                </div>
              )}
            </aside>
          )}

          {/* Main column */}
          <main className={mode === "agent" ? "md:col-span-2" : ""}>
            <ModeToggle
              mode={mode}
              onChange={handleModeChange}
              disabled={streaming}
            />

            {resetNotice && (
              <div className="mb-6 text-xs text-saffron-400 italic">
                {resetNotice}
              </div>
            )}

            {turns.length === 0 ? (
              <EmptyState mode={mode} />
            ) : (
              <div className="space-y-12">
                {turns.map((turn, idx) => (
                  <TurnView key={turn.id} turn={turn} index={idx} />
                ))}
              </div>
            )}

            <div ref={scrollAnchorRef} />
          </main>
        </div>
      </div>

      {/* Sticky chat input */}
      <div
        className="sticky bottom-0 left-0 right-0 border-t border-base-600 backdrop-blur"
        style={{ backgroundColor: "rgba(14, 9, 8, 0.92)" }}
      >
        <div className="max-w-6xl mx-auto px-8 py-5">
          <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] gap-10">
            {mode === "filtered" && <div className="hidden md:block" />}
            <div className={mode === "agent" ? "md:col-span-2" : ""}>
              <div className="relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={placeholder}
                  className="w-full pl-4 pr-24 py-3 bg-base-800 border border-base-600 rounded text-cream-100 placeholder:text-cream-500 focus:outline-none focus:border-saffron-400 focus:ring-1 focus:ring-saffron-400/50 transition"
                  disabled={streaming}
                  autoFocus
                />
                <button
                  onClick={handleSubmit}
                  disabled={streaming || !query.trim()}
                  className={`absolute right-1.5 top-1/2 -translate-y-1/2 px-4 py-1.5 text-sm font-medium rounded-sm transition ${
                    streaming || !query.trim()
                      ? "text-cream-500 cursor-not-allowed"
                      : "bg-saffron-400 text-base-900 hover:bg-saffron-400/90"
                  }`}
                >
                  {streaming ? "Thinking…" : "Ask"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ mode }) {
  return (
    <div className="text-cream-300 text-sm italic leading-relaxed max-w-md">
      {mode === "agent"
        ? "Ask anything about London shisha lounges. The agent will infer filters from your question and refine across follow-ups."
        : "Choose your filters in the sidebar and ask a question, or just describe what you're looking for."}
    </div>
  );
}

function TurnView({ turn, index }) {
  const {
    query,
    mode,
    parsed,
    statusMessage,
    streamingText,
    result,
    error,
    streaming,
  } = turn;
  const displayedText = result?.answer_validated ?? streamingText;
  const isDeclined = result?.is_declined === true;

  return (
    <div className="space-y-5">
      {/* The user's question, shown as a quiet prefix */}
      <div className="flex items-baseline gap-3">
        <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cream-500 shrink-0 mt-1">
          You · {index + 1}
        </span>
        <p className="text-cream-100 text-base leading-relaxed">{query}</p>
      </div>

      {/* Inferred filters - agent mode only, hidden on decline */}
      {mode === "agent" && parsed && !isDeclined && (
        <InferredFilters parsed={parsed} />
      )}

      {/* Answer card */}
      {(streamingText || result || statusMessage || error) && (
        <div
          className={`relative p-7 bg-base-800 rounded ${
            isDeclined
              ? "border border-base-600"
              : "border-l-2 border-l-saffron-400 border-y border-r border-y-base-600 border-r-base-600"
          }`}
        >
          <h2 className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cream-300 mb-4 flex items-center gap-2">
            {isDeclined ? "Outside system scope" : "Answer"}
            {streaming && !isDeclined && (
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-saffron-400 animate-pulse shadow-[0_0_8px_rgba(232,160,76,0.6)]" />
            )}
          </h2>
          {statusMessage && !isDeclined && (
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
      )}

      {/* Error */}
      {error && (
        <div className="p-4 bg-base-800 border border-terracotta-500/40 rounded text-terracotta-500 text-sm">
          <span className="font-semibold">Error:</span> {error}
        </div>
      )}

      {/* Evidence cards */}
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

      {/* Metadata footer - suppressed on decline */}
      {result && !isDeclined && (
        <div className="text-xs text-cream-500 pt-4 border-t border-base-600 space-y-3">
          <div className="flex flex-wrap gap-2">
            {result.degraded && (
              <div className="inline-block px-2.5 py-1 bg-base-800 border border-saffron-400/40 rounded text-saffron-400">
                Partial response — stream interrupted before completion.
              </div>
            )}
            {result.rerank_succeeded === false && (
              <div className="inline-block px-2.5 py-1 bg-base-800 border border-saffron-400/40 rounded text-saffron-400">
                Rerank unavailable — results ordered by similarity and recency
                only.
              </div>
            )}
          </div>
          <div className="font-mono tabular">
            {result.candidates_pulled} candidates pulled ·{" "}
            {result.chunks?.length || 0} chunks · {result.tokens_in} in /{" "}
            {result.tokens_out} out · ${result.cost_usd?.toFixed(4)} ·{" "}
            {result.quote_validations?.filter((v) => v.valid).length || 0}/
            {result.quote_validations?.length || 0} quotes verified
          </div>
        </div>
      )}
    </div>
  );
}

function ModeToggle({ mode, onChange, disabled }) {
  return (
    <div className="mb-6 inline-flex p-0.5 bg-base-800 border border-base-600 rounded">
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
  // Build pill list. lounge_focus comes first when set - it's the most
  // consequential filter (scopes to one venue).
  const pills = [];

  if (parsed.lounge_focus) {
    pills.push({
      key: "lounge_focus",
      label: "Focused on",
      value: LOUNGE_LABELS[parsed.lounge_focus] || parsed.lounge_focus,
      isLounge: true,
    });
  }
  if (parsed.area)
    pills.push({ key: "area", label: "Area", value: parsed.area });
  if (parsed.price_tier)
    pills.push({
      key: "price_tier",
      label: "Price",
      value: PRICE_LABELS[parsed.price_tier] || parsed.price_tier,
    });
  if (parsed.aspect_positive)
    pills.push({
      key: "aspect_positive",
      label: "Aspect",
      value: ASPECT_LABELS[parsed.aspect_positive] || parsed.aspect_positive,
    });

  // Set of filter keys that were carried forward from a prior turn.
  // Backend ships this as a sorted array; we use a Set for O(1) lookup.
  const inherited = new Set(parsed.inherited_filters || []);

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
        <div className="flex flex-wrap gap-2 items-center">
          {pills.map((p) => {
            const isInherited = inherited.has(p.key);
            const isFocus = p.isLounge;
            // Visual treatment:
            // - Focus pill gets a thin saffron border to mark it as dominant.
            // - Inherited pills are slightly dimmed and labelled.
            // - Fresh pills are the standard cream-on-base treatment.
            const baseClasses =
              "inline-flex items-center gap-1.5 px-3 py-1 bg-base-700 border rounded-sm text-xs transition";
            const variantClasses = isFocus
              ? "border-saffron-400/60"
              : isInherited
                ? "border-base-600 opacity-70"
                : "border-base-600";
            return (
              <span key={p.key} className={`${baseClasses} ${variantClasses}`}>
                <span className="text-cream-300">{p.label}:</span>
                <span className="font-medium text-cream-100">{p.value}</span>
                {isInherited && (
                  <span className="text-cream-500 italic ml-1">
                    · carried forward
                  </span>
                )}
              </span>
            );
          })}
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
