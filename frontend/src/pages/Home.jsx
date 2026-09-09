import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import {
  AlertCircle,
  BookOpen,
  Film,
  LogOut,
  Moon,
  Music,
  RefreshCw,
  Search,
  Sparkles,
  Sun,
} from 'lucide-react';

import { checkHealth, getRecommendations, getSimilarItems } from '../api/client';
import AISignalsPanel from '../components/AISignalsPanel';
import BackendStatus from '../components/BackendStatus';
import ContentFilter from '../components/ContentFilter';
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

const TYPE_META = [
  { label: 'Movies', value: 'movie', icon: Film, tint: 'var(--type-movie)' },
  { label: 'Books', value: 'book', icon: BookOpen, tint: 'var(--type-book)' },
  { label: 'Music', value: 'music', icon: Music, tint: 'var(--type-music)' },
];

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

  useEffect(() => {
    if (!hasRunDefault.current && backendStatus === 'online') {
      hasRunDefault.current = true;
      handleSearch();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendStatus]);

  async function handleSearch(nextQuery = query) {
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
        content_type: contentType,
      });
      const nextResults = data.results || [];
      setResults(nextResults);
      setElapsed(Math.round(performance.now() - startedAt));
      if (nextResults.length === 0) {
        addToast({ type: 'info', message: 'No results. Try a different query or filter.' });
      }
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Unknown error');
      setResults([]);
      setElapsed(null);
      addToast({
        type: 'error',
        title: 'Search failed',
        message: 'Backend returned an error. Check the API server.',
      });
    } finally {
      setLoading(false);
    }
  }

  function handleChipSelect(label) {
    setQuery(label);
    handleSearch(label);
  }

  async function handleSimilar(item) {
    setDrawer({ open: true, title: item.title, results: [], loading: true, error: null });
    try {
      const data = await getSimilarItems({
        global_id: item.global_id,
        user_id: userId,
        top_k: 10,
        content_type: null,
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
              style={{ background: 'linear-gradient(135deg, var(--accent), var(--accent-2))' }}
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
              <span className="chip mb-4" style={{ color: 'var(--accent)', borderColor: 'color-mix(in srgb, var(--accent) 30%, transparent)' }}>
                <Sparkles size={11} />
                Hybrid retrieval + graph re-ranking
              </span>
              <h1 className="display text-[2rem] font-extrabold leading-[1.08] text-ink sm:text-[2.6rem]">
                Search movies, books,
                <br />
                and music by meaning.
              </h1>
              <p className="mt-3 max-w-lg text-[13.5px] leading-relaxed text-ink-muted">
                Every result shows exactly which signals ranked it — semantic
                similarity, collaborative graph, session drift, and knowledge-graph
                proximity.
              </p>
            </div>

            {/* live counts per content type */}
            <div className="flex gap-2">
              {TYPE_META.map(({ label, value, icon: Icon, tint }) => {
                const count = results.filter((item) => item.content_type === value).length;
                return (
                  <div key={value} className="panel-flat min-w-[86px] px-3 py-2.5 text-center">
                    <Icon size={14} className="mx-auto" style={{ color: tint }} />
                    <p className="num display mt-1.5 text-xl font-bold tabular-nums text-ink">
                      {count}
                    </p>
                    <p className="text-[10px] font-medium text-ink-faint">{label}</p>
                  </div>
                );
              })}
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

            <GlassPanel className="p-4">
              <div className="mb-2.5 flex items-baseline justify-between">
                <p className="label">Results</p>
                <span className="num text-[11px] font-semibold tabular-nums text-accent">{topK}</span>
              </div>
              <input
                type="range"
                min={3}
                max={20}
                step={1}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
                className="mb-4 w-full cursor-pointer"
              />
              <p className="label mb-2">Content type</p>
              <ContentFilter value={contentType} onChange={setContentType} />
            </GlassPanel>

            <GlassPanel className="p-4">
              <p className="label mb-2">Try a query</p>
              <QueryChips onSelect={handleChipSelect} />
            </GlassPanel>

            <AISignalsPanel topResult={results[0] ?? null} />
          </aside>

          {/* ---------- results ---------- */}
          <section className="min-w-0">
            {backendStatus === 'offline' && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="panel mb-4 flex flex-col gap-3 p-5 sm:flex-row sm:items-center"
                style={{ borderColor: 'color-mix(in srgb, var(--bad) 35%, transparent)' }}
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
              <div className="panel px-6 py-20 text-center">
                <h3 className="display text-lg font-bold text-ink">No results found</h3>
                <p className="mt-2 text-[13px] text-ink-muted">
                  Try a different query or clear the content-type filter.
                </p>
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
