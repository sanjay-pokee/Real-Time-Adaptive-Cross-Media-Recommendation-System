import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  AlertCircle,
  ArrowUpRight,
  Factory,
  Film,
  HeartPulse,
  Landmark,
  LogOut,
  Moon,
  RefreshCw,
  ShieldAlert,
  SlidersHorizontal,
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
      // Key is versioned. The surface is light-first now, and a browser
      // holding the old "dark" preference would have opened the redesign in
      // the variant it was not designed around.
      return localStorage.getItem('nexus-theme-v3') || 'dark';
    } catch {
      return 'dark';
    }
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('nexus-theme-v3', theme);
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
  // The query the *displayed* results came from, which is not the same as
  // `query`: that tracks the input and changes on every keystroke, so the
  // results heading claimed to describe results it had nothing to do with
  // the moment someone typed without pressing Search.
  const [resultQuery, setResultQuery] = useState('');
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
  // Controls live off the page. Everything the old rail held is one
  // keystroke away, but none of it competes with the results for the
  // reader's attention - which is what a rail, horizontal or vertical,
  // always ends up doing.
  const [refineOpen, setRefineOpen] = useState(false);

  // A control panel you cannot see is only safe if the page still says what it
  // is doing. This is the number of active narrowings, shown on the trigger.
  const activeFilters = [
    domain,
    contentType,
    age !== null ? 'age' : null,
    safeMode ? 'safe' : null,
  ].filter(Boolean).length;

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

  // The hero's pitch collapses as soon as a search has been run, so the results
  // — the actual product — start near the top of the page instead of below it.
  const heroCollapsed = searched;

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

  // No search runs on load. Firing the default query meant the product opened
  // mid-task - a result set for a question nobody asked - and the landing page
  // was never seen. `hasRunDefault` is kept because a chip or a domain click
  // still needs to know whether the first search has happened.
  useEffect(() => {
    if (backendStatus === 'online') hasRunDefault.current = false;
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
      setResultQuery(cleanQuery);
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
              type="button"
              onClick={() => setRefineOpen(true)}
              className="btn"
              title="Filters, audience and profile"
            >
              <SlidersHorizontal size={13} />
              Refine
              {activeFilters > 0 && (
                <span
                  className="num ml-0.5 rounded-full px-1.5 text-[10px] font-bold"
                  style={{ background: 'var(--accent)', color: 'var(--accent-ink)' }}
                >
                  {activeFilters}
                </span>
              )}
            </button>

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
        {/* lg-refract: the hero is the one surface large enough for the
            displacement lensing to read, and cheap enough to do it on once.
            See the .lg-refract rule in globals.css. */}
        <GlassPanel
          variant="strong"
          className={`lg-refract relative z-20 mb-5 transition-[padding] duration-500 ${
            heroCollapsed ? 'p-4 sm:p-5' : 'p-6 sm:p-8'
          }`}
        >
          {/* The pitch is worth a full screen exactly once. After a search has
              run it is dead weight: it held ~300px permanently and pushed the
              first result card 1,932px down the page - two and a half screens
              of scrolling before a reviewer sees a single recommendation, on a
              product whose entire point is the recommendations. It collapses
              into the search row instead, and the tallies come with it. */}
          {/* Driven by `animate`, not by mount/unmount inside AnimatePresence.
              The exit variant only settles if its completion callback fires,
              and under StrictMode it does not always - which left the wrapper
              stranded at 49px around a 138px headline, so the top of "Search
              anything" bled through the search bar. Animating a persistent
              element to height 0 cannot stall: the target is the truth. */}
          <motion.div
            initial={false}
            animate={heroCollapsed
              ? { opacity: 0, height: 0 }
              : { opacity: 1, height: 'auto' }}
            transition={{ duration: 0.42, ease: [0.22, 1, 0.36, 1] }}
            aria-hidden={heroCollapsed}
            className="overflow-hidden"
          >
                <div className="max-w-3xl pb-9">
                  <span className="chip mb-6" style={{ color: 'var(--accent)', borderColor: 'color-mix(in oklab, var(--accent) 30%, transparent)' }}>
                    <Sparkles size={11} />
                    Hybrid retrieval + graph re-ranking
                  </span>
                  {/* Editorial scale: the headline is the page's one piece of
                      real typography, so it is set large enough to carry the
                      whitespace around it rather than sitting inside it. */}
                  <h1 className="display text-[2.9rem] leading-[0.98] text-ink sm:text-[4.4rem]">
                    Search anything
                    <br />
                    <em>by meaning,</em> not keyword.
                  </h1>
                  <p className="mt-6 max-w-xl text-[15px] leading-relaxed text-ink-muted">
                    One engine across {domainCount} domains. Every result shows the
                    signals that ranked it, and every result is checked against the
                    viewer before it is scored.
                  </p>
                </div>
          </motion.div>

          <div className={heroCollapsed ? '' : 'mt-0'}>
            <SearchBar
              value={query}
              onChange={setQuery}
              onSearch={(value) => handleSearch(value)}
              loading={loading}
              disabled={backendStatus === 'offline'}
            />
          </div>

          {/* Live counts, one tile per domain actually present in the results.
              Previously a fixed Movies/Books/Music trio, which showed three
              zeroes the moment you searched a health or finance query. Sits
              under the search row once collapsed so it stays visible without
              costing a band of its own. */}
          {domainTallies.length > 0 && (
            <motion.div layout className="mt-3 flex flex-wrap gap-2">
              {domainTallies.map(({ name, label, count, tint, icon: Icon }) => (
                <motion.div
                  key={name}
                  layout
                  className={`panel-flat flex items-center gap-2 ${
                    heroCollapsed ? 'px-2.5 py-1.5' : 'min-w-[86px] px-3 py-2.5'
                  }`}
                >
                  <Icon size={13} style={{ color: tint }} />
                  <p className="num display text-[15px] font-bold tabular-nums text-ink">
                    {count}
                  </p>
                  <p className="text-[10px] font-medium text-ink-faint">{label}</p>
                </motion.div>
              ))}
            </motion.div>
          )}
        </GlassPanel>

        {/* ================= body ================= */}
        {/* Two columns from `lg` (1024px), not `xl` (1280px). Below the split
            the filter rail stacks *above* the results, so at 1024-1279px — an
            ordinary laptop, and the width this gets demoed at — every filter
            panel came before the first recommendation. The rail is sticky, so
            the filters stay reachable without scrolling back up. */}
        <div className="flex flex-col gap-7">
          {/* ---------- sidebar ---------- */}
          <AnimatePresence>
            {refineOpen && (
              <>
                <motion.button
                  type="button"
                  aria-label="Close refine panel"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  onClick={() => setRefineOpen(false)}
                  className="fixed inset-0 z-40 cursor-default"
                  style={{ background: 'oklch(22% 0.02 62 / 0.34)' }}
                />
                <motion.aside
                  initial={{ x: '100%' }}
                  animate={{ x: 0 }}
                  exit={{ x: '100%' }}
                  transition={{ duration: 0.34, ease: [0.22, 1, 0.36, 1] }}
                  className="fixed right-0 top-0 z-50 flex h-full w-full max-w-[430px] flex-col gap-3 overflow-y-auto p-4"
                  style={{ background: 'var(--bg)', borderLeft: '1px solid var(--line)' }}
                >
                  <div className="mb-1 flex items-center justify-between">
                    <p className="display text-[26px]">Refine</p>
                    <button type="button" className="btn btn-ghost" onClick={() => setRefineOpen(false)}>
                      Done
                    </button>
                  </div>
            {/* One rail, not six floating cards.
                Each control used to be its own GlassPanel, so the sidebar was
                six stacked glass boxes with six rims, six blurs and six
                shadows - visual repetition that read as clutter, and six of the
                page's backdrop-filter layers spent on chrome rather than
                content. It is now a single glass surface with hairline-divided
                sections, which is both quieter and cheaper. */}
            <GlassPanel
              variant="strong"
              className="flex flex-col divide-y divide-line p-0"
            >
              <section className="p-4">
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
              </section>

              <section className="p-4">
              <p className="label mb-2.5">Domain</p>
              <DomainSelector
                domains={domainCatalog.domains || []}
                value={domain}
                onChange={setDomain}
              />
              </section>

              <section className="p-4">
              <p className="label mb-3">Audience</p>
              <AudienceControls
                age={age}
                onAgeChange={setAge}
                safeMode={safeMode}
                onSafeModeChange={setSafeMode}
              />
              </section>

              <section className="p-4">
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
              </section>

              <section className="p-4">
              <p className="label mb-2">Try a query</p>
              <QueryChips onSelect={handleChipSelect} />
              </section>
            </GlassPanel>

                  <AISignalsPanel topResult={results[0] ?? null} />
                </motion.aside>
              </>
            )}
          </AnimatePresence>

          {/* ---------- results ---------- */}
          {/* order-1 below `lg`: under the two-column split the rail stacks,
              and stacking it first put every filter panel ahead of the first
              recommendation again - 5.7 screens of scrolling at phone width.
              Results lead; the filters follow. */}
          <section className="order-2 min-w-0">
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
                    Cannot reach http://127.0.0.1:8000. Nothing here is broken — the
                    API is not answering.
                  </p>
                  {/* The command, not a description of it. This panel is most
                      likely to be read mid-demo, where the useful thing is
                      something to run rather than something to diagnose. */}
                  <code className="mt-2 block select-all rounded-lg bg-surface-3 px-2.5 py-1.5 font-mono text-[11px] text-ink-muted">
                    .venv\Scripts\python.exe -m uvicorn backend.app:app --port 8000
                  </code>
                  <p className="mt-1.5 text-[11px] text-ink-faint">
                    Takes about 40 s to load the embedding model. This panel clears
                    itself once /health answers.
                  </p>
                </div>
                <button onClick={pingBackend} className="btn btn-ghost px-3 py-2">
                  <RefreshCw size={13} />
                  Retry
                </button>
              </motion.div>
            )}

            {!loading && results.length > 0 && (
              <div className="mb-7 flex flex-wrap items-end justify-between gap-3 border-b pb-5"
                   style={{ borderColor: 'var(--line)' }}>
                <div className="min-w-0">
                  <p className="label">
                    {results.length} results
                    {elapsed !== null && <span className="ml-2 normal-case">· {elapsed} ms</span>}
                  </p>
                  {/* The heading is the query the *displayed* results came
                      from, so it is set as a masthead rather than a caption. */}
                  <h2 className="display mt-2 truncate text-[2rem] leading-tight text-ink sm:text-[2.6rem]">
                    {resultQuery}
                  </h2>
                </div>
              </div>
            )}

            {loading && (
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-2 2xl:grid-cols-3">
                {Array.from({ length: Math.min(topK, 6) }).map((_, index) => (
                  <SkeletonCard key={index} />
                ))}
              </div>
            )}

            {error && !loading && (
              <div className="panel px-6 py-10 text-center">
                <AlertCircle size={24} className="mx-auto" style={{ color: 'var(--bad)' }} />
                <p className="display mt-3 text-lg font-bold" style={{ color: 'var(--bad)' }}>
                  Search error
                </p>
                {/* Say what it means, then show the raw text. A timeout and a
                    dead API are different problems with different fixes, and
                    "timeout of 15000ms exceeded" alone tells a viewer neither. */}
                <p className="mx-auto mt-2 max-w-md text-[13px] leading-relaxed text-ink-muted">
                  {/timeout/i.test(String(error))
                    ? 'The API did not answer within 15 seconds. The first search after the server starts is slower, because the embedding model loads on demand — retrying usually succeeds.'
                    : 'The request reached the frontend but the API could not complete it. Retry, and if it persists the backend has most likely stopped.'}
                </p>
                <p className="mx-auto mt-3 max-w-md break-words font-mono text-[11px] text-ink-faint">
                  {error}
                </p>
                <button onClick={() => handleSearch()} className="btn btn-primary mx-auto mt-5 px-4 py-2">
                  <RefreshCw size={13} />
                  Retry
                </button>
              </div>
            )}

            {/* ================= homepage =================
                Shown until the first search. Nothing runs on load any more,
                so this is the product's actual front door rather than a
                placeholder behind an auto-fired query. It does three jobs:
                say what the engine is, show the verticals it covers as
                something you can enter, and hand over real queries to try. */}
            {!loading && !searched && backendStatus !== 'offline' && (
              <div className="flex flex-col gap-10">
                <section>
                  <div className="rule-capped mb-3" />
                  <p className="label">The verticals</p>
                  <div className="mt-5 grid grid-cols-1 gap-px sm:grid-cols-2 xl:grid-cols-4"
                       style={{ background: 'var(--line)' }}>
                    {(domainCatalog.domains || []).map((entry, i) => (
                      <button
                        key={entry.name}
                        type="button"
                        onClick={() => setDomain(entry.name)}
                        className="group relative flex flex-col items-start gap-2 p-6 text-left transition-colors"
                        style={{ background: 'var(--bg)' }}
                      >
                        <span
                          className="display text-[52px] leading-none"
                          style={{ color: 'var(--line-strong)' }}
                        >
                          {String(i + 1).padStart(2, '0')}
                        </span>
                        <span className="display text-[24px] leading-tight text-ink">
                          {entry.label}
                        </span>
                        <span className="text-[12.5px] leading-relaxed text-ink-muted">
                          {entry.description}
                        </span>
                        {entry.advisory && (
                          <span className="label mt-1" style={{ color: 'var(--warn)' }}>
                            Advisory applies
                          </span>
                        )}
                        <span
                          className="mt-2 h-0.5 w-0 transition-all duration-300 group-hover:w-12"
                          style={{ background: 'var(--accent)' }}
                        />
                      </button>
                    ))}
                  </div>
                </section>

                <section>
                  <div className="rule-capped mb-3" />
                  <p className="label">Try a query</p>
                  <p className="mt-3 max-w-xl text-[13.5px] leading-relaxed text-ink-muted">
                    A title, a person, a keyword or a plain description — retrieval is
                    hybrid, so all four are matched directly rather than only
                    semantically.
                  </p>
                  <div className="mt-5 flex flex-col" style={{ borderTop: '1px solid var(--line)' }}>
                    {[
                      ['Title', 'Breaking Bad'],
                      ['Person', 'Christopher Nolan'],
                      ['Keyword', 'time travel'],
                      ['Franchise', 'marvel cinematic universe'],
                      ['Description', 'space adventure with aliens'],
                      ['Vertical', 'how to invest my money'],
                    ].map(([kind, example]) => (
                      <button
                        key={kind}
                        type="button"
                        onClick={() => handleSearch(example)}
                        className="group flex items-baseline gap-5 py-3.5 text-left transition-colors"
                        style={{ borderBottom: '1px solid var(--line)' }}
                      >
                        <span className="label w-24 shrink-0">{kind}</span>
                        <span className="display flex-1 text-[21px] leading-snug text-ink transition-colors group-hover:text-accent">
                          {example}
                        </span>
                        <ArrowUpRight
                          size={15}
                          className="shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
                          style={{ color: 'var(--accent)' }}
                        />
                      </button>
                    ))}
                  </div>
                </section>
              </div>
            )}

            {/* Cross-fade between result sets.
                Keyed on the query the results came from, with mode="wait", so
                the outgoing set clears before the incoming one starts: the
                cards stagger in on their own 35 ms cascade, and running that
                over the top of the previous grid read as a flicker rather than
                a transition. The fade is deliberately short - 160 ms out - so
                it reads as responsive rather than as an animation to sit
                through, on a search that already takes ~400 ms to answer. */}
            <AnimatePresence mode="wait" initial={false}>
              {!loading && results.length > 0 && (
                <motion.div
                  key={resultQuery || 'results'}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.16, ease: 'easeOut' }}
                  className="flex flex-col gap-8"
                >
                  {/* Magazine hierarchy, not a uniform grid.
                      A ranked list whose first row looks exactly like its
                      tenth throws away the one thing the engine is asserting:
                      that #1 is the answer. The lead runs full width at a
                      reading measure; the rest fall into a three-up grid below
                      a rule, the way a section front works. */}
                  <RecommendationCard
                    key={results[0].global_id}
                    result={results[0]}
                    index={0}
                    variant="lead"
                    userId={userId}
                    query={query}
                    onSimilar={handleSimilar}
                    onView={setSelectedItem}
                    onToast={addToast}
                  />

                  {results.length > 1 && (
                    <>
                      <div>
                        <div className="rule-capped mb-3" />
                        <span className="label">More results</span>
                      </div>

                      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-3">
                        {results.slice(1).map((result, index) => (
                          <RecommendationCard
                            key={result.global_id}
                            result={result}
                            index={index + 1}
                            userId={userId}
                            query={query}
                            onSimilar={handleSimilar}
                            onView={setSelectedItem}
                            onToast={addToast}
                          />
                        ))}
                      </div>
                    </>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

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
                    {/* Name the constraints that are actually on. "Clear the
                        content-type filter" is unhelpful when the filter is not
                        set and the domain is - the reader is told to undo
                        something they never did, while the thing that bound
                        goes unmentioned. */}
                    <p className="mx-auto mt-2 max-w-md text-[13px] leading-relaxed text-ink-muted">
                      Nothing matched <span className="font-semibold text-ink">{resultQuery}</span>
                      {contentType || activeDomain ? ' within the current scope.' : '.'}
                    </p>
                    {(contentType || activeDomain) && (
                      <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
                        {activeDomain && (
                          <button
                            type="button"
                            onClick={() => setDomain(null)}
                            className="btn btn-ghost px-3 py-1.5 text-[12px]"
                          >
                            Clear domain: {activeDomain.label}
                          </button>
                        )}
                        {contentType && (
                          <button
                            type="button"
                            onClick={() => setContentType(null)}
                            className="btn btn-ghost px-3 py-1.5 text-[12px]"
                          >
                            Clear type: {contentType}
                          </button>
                        )}
                      </div>
                    )}
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
        // The whole result set, so the panel can re-rank with a signal removed
        // and show where this item would actually have landed.
        results={results}
        effectiveAge={effectiveAge}
      />
      <Toast toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
