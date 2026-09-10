import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import {
  AlertCircle,
  Factory,
  Film,
  HeartPulse,
  Landmark,
  LogOut,
  Moon,
  RefreshCw,
  Search,
  ShieldAlert,
  Sparkles,
  Sun,
} from 'lucide-react';

import { checkHealth, getDomains, getRecommendations, getSimilarItems } from '../api/client';
import AdvisoryBanner from '../components/AdvisoryBanner';
import AISignalsPanel from '../components/AISignalsPanel';
import AudienceControls from '../components/AudienceControls';
import BackendStatus from '../components/BackendStatus';
import ContentFilter from '../components/ContentFilter';
import DomainSelector from '../components/DomainSelector';
import GlassPanel from '../components/GlassPanel';
import ItemDetailModal from '../components/ItemDetailModal';
import QueryChips from '../components/QueryChips';
import RecommendationCard from '../components/RecommendationCard';
import SearchBar from '../components/SearchBar';
import SimilarDrawer from '../components/SimilarDrawer';
import SkeletonCard from '../components/SkeletonCard';
import Toast from '../components/Toast';
import { createToast } from '../utils/toast';
import UserSelector, { USERS } from '../components/UserSelector';

const DEFAULT_QUERY = 'space adventure with aliens';
const DEFAULT_TOP_K = 10;

// Short labels and icons for the hero counters, keyed by domain. A domain
// with no entry here still gets a tile, using its own label from /domains.
const DOMAIN_TILE = {
  entertainment: { label: 'Media',    icon: Film },
  health:        { label: 'Health',   icon: HeartPulse },
  industry:      { label: 'Industry', icon: Factory },
  finance:       { label: 'Finance',  icon: Landmark },
};

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('nexus-theme') || 'dark';
    } catch {
      return 'dark';
    }
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('nexus-theme', theme);
    } catch {
      /* storage can be unavailable (private mode) - the theme still applies */
    }
  }, [theme]);

  return [theme, () => setTheme((value) => (value === 'dark' ? 'light' : 'dark'))];
}

export default function Home({ authenticatedUser, onLogout }) {
  const availableUsers = useMemo(() => {
    if (!authenticatedUser || USERS.some((user) => user.id === authenticatedUser.id)) return USERS;
    return [authenticatedUser, ...USERS];
  }, [authenticatedUser]);

  const [theme, toggleTheme] = useTheme();
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [userId, setUserId] = useState(authenticatedUser?.id || USERS[0].id);
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [contentType, setContentType] = useState(null);
  // Audience + domain scope. `age: null` is a real state, not "unset": the
  // backend treats a request with no age as capped at the teen ceiling rather
  // than as an adult, so the UI has to be able to express it.
  const [domain, setDomain] = useState(null);
  const [age, setAge] = useState(null);
  const [safeMode, setSafeMode] = useState(false);
  const [domainCatalog, setDomainCatalog] = useState({ domains: [], content_types: [] });
  const [advisory, setAdvisory] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [toasts, setToasts] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);
  const [elapsed, setElapsed] = useState(null);
  const [drawer, setDrawer] = useState({
    open: false,
    title: '',
    results: [],
    loading: false,
    error: null,
  });
  const hasRunDefault = useRef(false);
  // Monotonic id of the newest search, so an out-of-order response can be
  // discarded rather than overwriting fresher results.
  const searchSeq = useRef(0);
  // Bumped to force a scope re-search when only the query text changed.
  const [scopeNonce, setScopeNonce] = useState(0);

  const activeUser = availableUsers.find((user) => user.id === userId) || availableUsers[0];
  const averageScore = results.length
    ? results.reduce((sum, item) => sum + Number(item.score || 0), 0) / results.length
    : 0;

  const addToast = useCallback((opts) => {
    const toast = createToast(opts.message, opts.type || 'info', opts.title || '', opts.duration);
    setToasts((prev) => [...prev.slice(-4), toast]);
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const pingBackend = useCallback(async () => {
    setBackendStatus('checking');
    try {
      await checkHealth();
      setBackendStatus('online');
    } catch {
      setBackendStatus('offline');
    }
  }, []);

  useEffect(() => {
    pingBackend();
  }, [pingBackend]);

  useEffect(() => {
    if (authenticatedUser?.id) setUserId(authenticatedUser.id);
  }, [authenticatedUser?.id]);

  // Load the domain packs once the backend answers. A failure here is not
  // fatal: the selector simply shows "All domains" and search still works.
  useEffect(() => {
    if (backendStatus !== 'online') return;
    let cancelled = false;
    getDomains()
      .then((data) => {
        if (!cancelled) setDomainCatalog(data);
      })
      .catch(() => {
        /* leave the catalog empty; the UI degrades to domain-agnostic search */
      });
    return () => { cancelled = true; };
  }, [backendStatus]);

  const activeDomain = useMemo(
    () => domainCatalog.domains?.find((entry) => entry.name === domain) || null,
    [domainCatalog.domains, domain],
  );

  const domainCount = domainCatalog.domains?.length || 0;

  // Distinguish "the query matched nothing" from "the eligibility stage
  // withheld everything". Both look like zero results, but only one is worth
  // explaining, and only one has an obvious fix.
  const effectiveAge = safeMode ? 7 : age ?? 13;
  const audienceBlocked = effectiveAge < 18 && (
    // A regulated domain is adult-only end to end.
    (activeDomain?.risk_tier ?? 0) >= 1
    // As is the industrial catalogue, which is fixed at adult.
    || activeDomain?.name === 'industry'
    || contentType === 'industrial'
  );

  // One tally per domain that actually appears in the current results, so the
  // hero counters follow the query instead of always naming the same three
  // entertainment types.
  const domainTallies = useMemo(() => {
    const counts = new Map();
    for (const item of results) {
      const key = item.domain || 'other';
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    return (domainCatalog.domains || [])
      .filter((entry) => counts.has(entry.name))
      .map((entry) => ({
        name: entry.name,
        label: DOMAIN_TILE[entry.name]?.label || entry.label,
        icon: DOMAIN_TILE[entry.name]?.icon || Sparkles,
        tint: `var(--dom-${entry.name})`,
        count: counts.get(entry.name),
      }));
  }, [results, domainCatalog.domains]);

  // The content types offered by the current scope: the selected domain's own
  // types, or every known type when no domain is chosen.
  const offeredContentTypes = useMemo(
    () => activeDomain?.content_types || domainCatalog.content_types || [],
    [activeDomain, domainCatalog.content_types],
  );

  // Switching domain can strand a content-type filter that the new domain does
  // not contain, which would silently return nothing. Clear it.
  useEffect(() => {
    if (contentType && !offeredContentTypes.includes(contentType)) {
      setContentType(null);
    }
  }, [offeredContentTypes, contentType]);

  // Re-run the search when the scope changes, so picking a domain or dragging
  // the age slider updates the results immediately instead of leaving stale
  // ones on screen until the user thinks to press Search again.
  //
  // Debounced because the age slider fires on every step; without it a drag
  // from 8 to 30 would issue twenty-odd requests. Skipped until the first
  // search has run, so it does not race the initial load.
  const searchRef = useRef(handleSearch);
  searchRef.current = handleSearch;

  useEffect(() => {
    if (!searched) return undefined;
    const timer = setTimeout(() => searchRef.current(query), 350);
    return () => clearTimeout(timer);
    // `query` is deliberately absent from the deps: typing should not
    // auto-search, only changing who and what we are searching for should.
    // `scopeNonce` lets a chip force a run when it changes the query but
    // happens to leave the scope identical.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [domain, age, safeMode, contentType, topK, scopeNonce]);

  useEffect(() => {
    if (!hasRunDefault.current && backendStatus === 'online') {
      hasRunDefault.current = true;
      handleSearch();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendStatus]);

  /**
   * `overrides` exists because a query chip sets the domain and the age and
   * then searches in the same gesture. setState is async, so reading those
   * off state here would use the previous values and run the wrong search.
   */
  async function handleSearch(nextQuery = query, overrides = {}) {
    const scope = {
      domain: 'domain' in overrides ? overrides.domain : domain,
      age: 'age' in overrides ? overrides.age : age,
      safeMode: 'safeMode' in overrides ? overrides.safeMode : safeMode,
      contentType: 'contentType' in overrides ? overrides.contentType : contentType,
    };
    // Searches can overlap — a chip changes several bits of scope at once and
    // the slider fires repeatedly — and responses do not necessarily come back
    // in the order they were sent. Without this guard a slower earlier request
    // could land last and overwrite the correct results with stale ones.
    const seq = ++searchSeq.current;
    const isStale = () => seq !== searchSeq.current;
    const cleanQuery = nextQuery.trim();
    if (!cleanQuery) return;
    setQuery(cleanQuery);
    setLoading(true);
    setError(null);
    setSearched(true);
    const startedAt = performance.now();
    try {
      const data = await getRecommendations({
        query: cleanQuery,
        user_id: userId,
        top_k: topK,
        content_type: scope.contentType,
        domain: scope.domain,
        age: scope.age,
        safe_mode: scope.safeMode,
      });
      if (isStale()) return;
      const nextResults = data.results || [];
      setResults(nextResults);
      // The advisory is a property of the response, not of the selector: the
      // backend decides whether this particular result set needs one.
      setAdvisory(data.advisory || null);
      setElapsed(Math.round(performance.now() - startedAt));
      // Silent when the audience filter is the cause: the empty state already
      // explains that case in full, and a toast on top of it just nags.
      if (nextResults.length === 0 && !audienceBlocked) {
        addToast({ type: 'info', message: 'No results. Try a different query or filter.' });
      }
    } catch (err) {
      if (isStale()) return;
      setError(err?.response?.data?.detail || err.message || 'Unknown error');
      setResults([]);
      setElapsed(null);
      addToast({
        type: 'error',
        title: 'Search failed',
        message: 'Backend returned an error. Check the API server.',
      });
    } finally {
      if (!isStale()) setLoading(false);
    }
  }

  /**
   * A chip is a whole scenario: query, domain and viewer age together. It
   * clears any content-type filter, which would otherwise survive from a
   * previous domain and silently empty the results.
   */
  function handleChipSelect(chip) {
    setQuery(chip.label);
    setDomain(chip.domain);
    setAge(chip.age);
    setSafeMode(false);
    setContentType(null);
    // Deliberately does NOT call handleSearch itself. React batches these into
    // one re-render and the scope effect then runs a single search with all of
    // them applied. Calling it here as well fired a second request from the
    // pre-update render — domain already finance, age still null — which
    // returned nothing and, resolving last, won.
    setScopeNonce((value) => value + 1);
  }

  async function handleSimilar(item) {
    setDrawer({ open: true, title: item.title, results: [], loading: true, error: null });
    try {
      // The audience scope has to travel with this call too. Without it the
      // drawer was a hole straight through the eligibility layer: a viewer set
      // to 8 could click "similar" on a childrens' film and get adult items
      // back, because /recommend/item was being asked with no age at all.
      // Domain is deliberately not forwarded — "more like this" legitimately
      // crosses verticals, and that is the cross-media claim. Age is not
      // optional in the same way.
      const data = await getSimilarItems({
        global_id: item.global_id,
        user_id: userId,
        top_k: 10,
        content_type: null,
        age,
        safe_mode: safeMode,
      });
      setDrawer((current) => ({ ...current, results: data.results || [], loading: false }));
    } catch (err) {
      setDrawer((current) => ({ ...current, loading: false, error: err.message }));
      addToast({ type: 'error', title: 'Similar search failed', message: err.message });
    }
  }

  return (
    <div className="above min-h-screen">
      {/* ================= header ================= */}
      <header className="sticky top-0 z-40 border-b border-line bg-bg/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1500px] items-center gap-3 px-5 py-3">
          <div className="flex items-center gap-2.5">
            <div
              className="flex h-8 w-8 items-center justify-center rounded-xl text-white"
              style={{ background: 'linear-gradient(135deg in oklab, var(--accent), var(--accent-2))' }}
            >
              <Sparkles size={15} />
            </div>
            <div className="leading-none">
              <p className="display text-[15px] font-bold text-ink">Nexus</p>
              <p className="mt-1 hidden text-[9px] font-semibold uppercase tracking-[0.14em] text-ink-faint sm:block">
                Cross-media discovery
              </p>
            </div>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <BackendStatus status={backendStatus} onRetry={pingBackend} />

            <div className="hidden items-center gap-2 rounded-xl border border-line bg-surface-3 px-2 py-1.5 md:flex">
              <span
                className="flex h-6 w-6 items-center justify-center rounded-lg text-[9px] font-bold text-white"
                style={{ background: activeUser.accent }}
              >
                {activeUser.initials}
              </span>
              <span className="text-[12px] font-semibold text-ink">{activeUser.label}</span>
            </div>

            <button
              onClick={toggleTheme}
              className="btn btn-ghost h-8 w-8 p-0"
              title={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}
            >
              {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
            </button>

            <button onClick={onLogout} className="btn btn-ghost h-8 w-8 p-0" title="Log out">
              <LogOut size={14} />
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1500px] px-5 py-6">
        {/* ================= hero + search ================= */}
        {/* No overflow-hidden here: the search bar's autocomplete dropdown is
            absolutely positioned and would be clipped by this panel's edge. */}
        <GlassPanel variant="strong" className="relative z-20 mb-5 p-6 sm:p-8">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-2xl">
              <span className="chip mb-4" style={{ color: 'var(--accent)', borderColor: 'color-mix(in oklab, var(--accent) 30%, transparent)' }}>
                <Sparkles size={11} />
                Hybrid retrieval + graph re-ranking
              </span>
              <h1 className="display text-[2rem] font-extrabold leading-[1.08] text-ink sm:text-[2.6rem]">
                Search anything
                <br />
                <span className="text-aurora">by meaning, not keyword.</span>
              </h1>
              <p className="mt-3 max-w-lg text-[13.5px] leading-relaxed text-ink-muted">
                One engine across {domainCount} domains. Every result shows the
                signals that ranked it, and every result is checked against the
                viewer before it is scored.
              </p>
            </div>

            {/* Live counts, one tile per domain actually present in the
                results. Previously a fixed Movies/Books/Music trio, which
                showed three zeroes the moment you searched a health or
                finance query. */}
            <div className="flex flex-wrap gap-2">
              {domainTallies.map(({ name, label, count, tint, icon: Icon }) => (
                <div key={name} className="panel-flat min-w-[86px] px-3 py-2.5 text-center">
                  <Icon size={14} className="mx-auto" style={{ color: tint }} />
                  <p className="num display mt-1.5 text-xl font-bold tabular-nums text-ink">
                    {count}
                  </p>
                  <p className="text-[10px] font-medium text-ink-faint">{label}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-6">
            <SearchBar
              value={query}
              onChange={setQuery}
              onSearch={(value) => handleSearch(value)}
              loading={loading}
              disabled={backendStatus === 'offline'}
            />
          </div>
        </GlassPanel>

        {/* ================= body ================= */}
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[264px_1fr]">
          {/* ---------- sidebar ---------- */}
          <aside className="flex flex-col gap-3 xl:sticky xl:top-[76px] xl:self-start">
            <GlassPanel className="p-4">
              <p className="label mb-2.5">Profile</p>
              <UserSelector value={userId} onChange={setUserId} users={availableUsers} />

              <div className="mt-3 grid grid-cols-2 gap-2">
                <div className="panel-flat px-3 py-2">
                  <p className="text-[10px] text-ink-faint">Results</p>
                  <p className="num display mt-0.5 text-lg font-bold tabular-nums text-ink">
                    {results.length}
                  </p>
                </div>
                <div className="panel-flat px-3 py-2">
                  <p className="text-[10px] text-ink-faint">Avg score</p>
                  <p className="num display mt-0.5 text-lg font-bold tabular-nums text-ink">
                    {averageScore.toFixed(2)}
                  </p>
                </div>
              </div>
            </GlassPanel>

            <GlassPanel variant="strong" className="p-4">
              <p className="label mb-2.5">Domain</p>
              <DomainSelector
                domains={domainCatalog.domains || []}
                value={domain}
                onChange={setDomain}
              />
            </GlassPanel>

            <GlassPanel variant="strong" className="p-4">
              <p className="label mb-3">Audience</p>
              <AudienceControls
                age={age}
                onAgeChange={setAge}
                safeMode={safeMode}
                onSafeModeChange={setSafeMode}
              />
            </GlassPanel>

            <GlassPanel className="p-4">
              <div className="mb-2.5 flex items-baseline justify-between">
                <p className="label">Results</p>
                <span className="num text-[11px] font-semibold tabular-nums text-accent">{topK}</span>
              </div>
              <input
                type="range"
                className="slider mb-4"
                min={3}
                max={20}
                step={1}
                value={topK}
                aria-label="Number of results"
                style={{ '--fill': `${Math.round(((topK - 3) / 17) * 100)}%` }}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
              <p className="label mb-2">Content type</p>
              <ContentFilter
                value={contentType}
                onChange={setContentType}
                contentTypes={offeredContentTypes}
              />
            </GlassPanel>

            <GlassPanel className="p-4">
              <p className="label mb-2">Try a query</p>
              <QueryChips onSelect={handleChipSelect} />
            </GlassPanel>

            <AISignalsPanel topResult={results[0] ?? null} />
          </aside>

          {/* ---------- results ---------- */}
          <section className="min-w-0">
            {/* Regulated domains return an advisory with every response. It
                sits above the results, not inside a card, because it governs
                the whole set. */}
            <AdvisoryBanner
              advisory={advisory}
              tint={
                activeDomain?.name
                  ? `var(--dom-${activeDomain.name})`
                  : 'var(--warn)'
              }
            />

            {backendStatus === 'offline' && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="panel mb-4 flex flex-col gap-3 p-5 sm:flex-row sm:items-center"
                style={{ borderColor: 'color-mix(in oklab, var(--bad) 35%, transparent)' }}
              >
                <AlertCircle size={20} style={{ color: 'var(--bad)' }} className="shrink-0" />
                <div className="flex-1">
                  <h2 className="display text-[15px] font-bold text-ink">Backend offline</h2>
                  <p className="mt-1 text-xs text-ink-muted">
                    Cannot reach http://127.0.0.1:8000 — start the FastAPI server to load results.
                  </p>
                </div>
                <button onClick={pingBackend} className="btn btn-ghost px-3 py-2">
                  <RefreshCw size={13} />
                  Retry
                </button>
              </motion.div>
            )}

            {!loading && results.length > 0 && (
              <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
                <div className="min-w-0">
                  <p className="label">
                    {results.length} results
                    {elapsed !== null && <span className="ml-2 normal-case">· {elapsed} ms</span>}
                  </p>
                  <h2 className="display mt-1 truncate text-xl font-bold text-ink">{query}</h2>
                </div>
              </div>
            )}

            {loading && (
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 2xl:grid-cols-3">
                {Array.from({ length: Math.min(topK, 6) }).map((_, index) => (
                  <SkeletonCard key={index} />
                ))}
              </div>
            )}

            {error && !loading && (
              <div className="panel p-8 text-center">
                <p className="display font-bold" style={{ color: 'var(--bad)' }}>
                  Search error
                </p>
                <p className="mt-2 font-mono text-xs text-ink-muted">{error}</p>
                <button onClick={() => handleSearch()} className="btn btn-primary mx-auto mt-4 px-4 py-2">
                  <RefreshCw size={13} />
                  Retry
                </button>
              </div>
            )}

            {!loading && !searched && backendStatus !== 'offline' && (
              <div className="panel px-6 py-20 text-center">
                <Search size={28} className="mx-auto text-ink-faint" />
                <h2 className="display mt-4 text-xl font-bold text-ink">Start with a search</h2>
                <p className="mx-auto mt-2 max-w-sm text-[13px] text-ink-muted">
                  Describe a mood, genre, scene, or artist style — not just a keyword.
                </p>
              </div>
            )}

            {!loading && results.length > 0 && (
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 2xl:grid-cols-3">
                {results.map((result, index) => (
                  <RecommendationCard
                    key={result.global_id}
                    result={result}
                    index={index}
                    userId={userId}
                    query={query}
                    onSimilar={handleSimilar}
                    onView={setSelectedItem}
                    onToast={addToast}
                  />
                ))}
              </div>
            )}

            {!loading && searched && results.length === 0 && !error && backendStatus !== 'offline' && (
              <div className="panel px-6 py-16 text-center">
                {audienceBlocked ? (
                  <>
                    {/* An empty result set caused by the eligibility stage is
                        not a failure, it is the constraint layer working. Say
                        which rule bound, otherwise it reads as a broken query. */}
                    <span
                      className="chip chip-tinted mx-auto mb-3"
                      style={{ '--tint': 'var(--mat-adult)' }}
                    >
                      <ShieldAlert size={11} />
                      Withheld by the audience filter
                    </span>
                    <h3 className="display text-lg font-bold text-ink">
                      Nothing here is eligible for this viewer
                    </h3>
                    <p className="mx-auto mt-2 max-w-md text-[13px] leading-relaxed text-ink-muted">
                      {activeDomain
                        ? `Every item in ${activeDomain.label} requires 18+. `
                        : 'The matching items require a higher age than this viewer has. '}
                      {safeMode
                        ? 'Safe mode is on, which caps below an adult age.'
                        : age === null
                          ? 'No age was declared, so the engine capped at the teen ceiling rather than assuming an adult.'
                          : `The declared age is ${age}.`}
                    </p>
                    <button
                      type="button"
                      className="btn btn-primary mt-5 px-4"
                      onClick={() => { setSafeMode(false); setAge(30); }}
                    >
                      Search as a 30-year-old
                    </button>
                  </>
                ) : (
                  <>
                    <h3 className="display text-lg font-bold text-ink">No results found</h3>
                    <p className="mt-2 text-[13px] text-ink-muted">
                      Try a different query or clear the content-type filter.
                    </p>
                  </>
                )}
              </div>
            )}
          </section>
        </div>
      </main>

      <SimilarDrawer
        open={drawer.open}
        onClose={() => setDrawer((current) => ({ ...current, open: false }))}
        title={drawer.title}
        results={drawer.results}
        loading={drawer.loading}
        error={drawer.error}
        userId={userId}
        query={query}
        onSimilar={handleSimilar}
        onView={setSelectedItem}
        onToast={addToast}
      />

      <ItemDetailModal
        item={selectedItem}
        onClose={() => setSelectedItem(null)}
        userId={userId}
        query={query}
        onSimilar={handleSimilar}
        onToast={addToast}
      />
      <Toast toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
