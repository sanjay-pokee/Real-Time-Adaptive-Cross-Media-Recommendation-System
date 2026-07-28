import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { AlertCircle, BookOpen, Film, Layers, LogOut, Music, RefreshCw, Search, Settings2, Sparkles, UserRound } from 'lucide-react';

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
  { label: 'Movies', value: 'movie', icon: Film },
  { label: 'Books', value: 'book', icon: BookOpen },
  { label: 'Music', value: 'music', icon: Music },
];

export default function Home({ authenticatedUser, onLogout }) {
  const availableUsers = useMemo(() => {
    if (!authenticatedUser || USERS.some(user => user.id === authenticatedUser.id)) return USERS;
    return [authenticatedUser, ...USERS];
  }, [authenticatedUser]);

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
  const [drawer, setDrawer] = useState({ open: false, title: '', results: [], loading: false, error: null });
  const hasRunDefault = useRef(false);

  const activeUser = availableUsers.find(user => user.id === userId) || availableUsers[0];
  const totalScore = results.reduce((sum, item) => sum + Number(item.score || 0), 0);
  const averageScore = results.length ? totalScore / results.length : 0;

  const addToast = useCallback((opts) => {
    const toast = createToast(opts.message, opts.type || 'info', opts.title || '', opts.duration);
    setToasts(prev => [...prev.slice(-4), toast]);
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
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

  useEffect(() => { pingBackend(); }, [pingBackend]);

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
    try {
      const data = await getRecommendations({ query: cleanQuery, user_id: userId, top_k: topK, content_type: contentType });
      const nextResults = data.results || [];
      setResults(nextResults);
      if (nextResults.length === 0) addToast({ type: 'info', message: 'No results found. Try changing the query or filter.' });
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Unknown error');
      setResults([]);
      addToast({ type: 'error', title: 'Search failed', message: 'Backend returned an error. Check the API server.' });
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
      const data = await getSimilarItems({ global_id: item.global_id, user_id: userId, top_k: 10, content_type: null });
      setDrawer(current => ({ ...current, results: data.results || [], loading: false }));
    } catch (err) {
      setDrawer(current => ({ ...current, loading: false, error: err.message }));
      addToast({ type: 'error', title: 'Similar search failed', message: err.message });
    }
  }

  return (
    <div className="min-h-screen bg-app">
      <header className="sticky top-0 z-40 border-b border-slate-200/80 bg-white/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-950 text-white shadow-sm">
              <Sparkles size={18} />
            </div>
            <div>
              <p className="font-display text-lg font-black leading-none tracking-tight text-slate-950">Nexus</p>
              <p className="mt-1 hidden text-xs font-bold uppercase tracking-[0.16em] text-slate-400 sm:block">Recommendation suite</p>
            </div>
          </div>

          <div className="ml-auto flex items-center gap-3">
            <BackendStatus status={backendStatus} onRetry={pingBackend} />
            <div className="hidden items-center gap-2 rounded-2xl border border-slate-200 bg-white px-3 py-2 shadow-sm md:flex">
              <span className="flex h-8 w-8 items-center justify-center rounded-xl text-xs font-black text-white" style={{ background: activeUser.accent }}>
                {activeUser.initials}
              </span>
              <div className="leading-tight">
                <p className="text-sm font-extrabold text-slate-900">{activeUser.label}</p>
                <p className="text-xs font-semibold text-slate-400">{activeUser.id}</p>
              </div>
            </div>
            <button onClick={onLogout} className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-500 transition hover:border-red-200 hover:text-red-600" title="Log out">
              <LogOut size={17} />
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:py-8">
        <section className="mb-6 grid gap-4 lg:grid-cols-[1fr_320px]">
          <GlassPanel variant="strong" className="overflow-hidden p-5 sm:p-7">
            <div className="flex flex-col gap-6">
              <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
                <div className="max-w-2xl">
                  <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-blue-100 bg-blue-50 px-3 py-1 text-xs font-extrabold text-blue-700">
                    <Layers size={13} />
                    Hybrid discovery
                  </div>
                  <h1 className="font-display text-3xl font-black tracking-tight text-slate-950 sm:text-4xl">Search across movies, books, and music.</h1>
                  <p className="mt-3 max-w-xl text-sm leading-6 text-slate-500">Use natural language and compare the semantic, graph, EMA, and final ranking signals behind each recommendation.</p>
                </div>
                <div className="grid grid-cols-3 gap-2 rounded-2xl bg-slate-50 p-2">
                  {TYPE_META.map(({ label, value, icon: Icon }) => {
                    const count = results.filter(item => item.content_type === value).length;
                    return (
                      <div key={value} className="rounded-xl bg-white px-3 py-2 text-center shadow-sm">
                        <Icon size={16} className="mx-auto text-slate-500" />
                        <p className="mt-1 text-lg font-black text-slate-950">{count}</p>
                        <p className="text-[11px] font-bold text-slate-400">{label}</p>
                      </div>
                    );
                  })}
                </div>
              </div>

              <SearchBar value={query} onChange={setQuery} onSearch={() => handleSearch()} loading={loading} disabled={backendStatus === 'offline'} />
            </div>
          </GlassPanel>

          <GlassPanel className="p-5">
            <div className="mb-4 flex items-center gap-2 text-sm font-black text-slate-900">
              <UserRound size={17} className="text-blue-600" />
              Active profile
            </div>
            <UserSelector value={userId} onChange={setUserId} users={availableUsers} />
            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="rounded-2xl bg-slate-50 p-3">
                <p className="text-xs font-bold text-slate-400">Results</p>
                <p className="mt-1 text-2xl font-black text-slate-950">{results.length}</p>
              </div>
              <div className="rounded-2xl bg-slate-50 p-3">
                <p className="text-xs font-bold text-slate-400">Avg score</p>
                <p className="mt-1 text-2xl font-black text-slate-950">{averageScore.toFixed(2)}</p>
              </div>
            </div>
          </GlassPanel>
        </section>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[320px_1fr]">
          <aside className="flex flex-col gap-4 xl:sticky xl:top-[88px] xl:self-start">
            <GlassPanel className="p-5">
              <div className="mb-4 flex items-center gap-2 text-sm font-black text-slate-900">
                <Settings2 size={17} className="text-blue-600" />
                Search controls
              </div>
              <label className="mb-2 block text-xs font-extrabold uppercase tracking-[0.12em] text-slate-400">Results: {topK}</label>
              <input
                type="range"
                min={3}
                max={20}
                step={1}
                value={topK}
                onChange={event => setTopK(Number(event.target.value))}
                className="mb-5 w-full cursor-pointer"
              />
              <label className="mb-2 block text-xs font-extrabold uppercase tracking-[0.12em] text-slate-400">Content type</label>
              <ContentFilter value={contentType} onChange={setContentType} />
            </GlassPanel>

            <GlassPanel className="p-5">
              <p className="mb-3 text-sm font-black text-slate-900">Popular searches</p>
              <QueryChips onSelect={handleChipSelect} />
            </GlassPanel>

            <AISignalsPanel userId={userId} backendStatus={backendStatus} topResult={results[0] ?? null} />
          </aside>

          <section className="min-w-0">
            {backendStatus === 'offline' && (
              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="mb-5 rounded-3xl border border-red-200 bg-red-50 p-5 text-red-900">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
                  <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white text-red-600 shadow-sm">
                    <AlertCircle size={22} />
                  </div>
                  <div className="flex-1">
                    <h2 className="font-display text-lg font-black">Backend offline</h2>
                    <p className="mt-1 text-sm font-medium text-red-700">Cannot reach http://127.0.0.1:8000. Start FastAPI to load live recommendations.</p>
                  </div>
                  <button onClick={pingBackend} className="flex items-center justify-center gap-2 rounded-xl bg-white px-4 py-2.5 text-sm font-extrabold text-red-700 shadow-sm transition hover:bg-red-100">
                    <RefreshCw size={15} />
                    Retry
                  </button>
                </div>
              </motion.div>
            )}

            {!loading && !searched && backendStatus !== 'offline' && (
              <div className="rounded-3xl border border-dashed border-slate-300 bg-white/70 px-6 py-16 text-center">
                <Search size={34} className="mx-auto text-slate-300" />
                <h2 className="mt-4 font-display text-2xl font-black text-slate-950">Start with a search</h2>
                <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-500">Try a mood, genre, scene, learning goal, or artist style.</p>
              </div>
            )}

            {loading && (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {Array.from({ length: Math.min(topK, 6) }).map((_, index) => <SkeletonCard key={index} />)}
              </div>
            )}

            {error && !loading && (
              <div className="rounded-3xl border border-red-200 bg-white p-6 text-center shadow-sm">
                <p className="font-black text-red-600">Search error</p>
                <p className="mt-2 text-sm font-mono text-slate-500">{error}</p>
                <button onClick={() => handleSearch()} className="btn-primary mx-auto mt-4 flex items-center gap-2 px-4 py-2.5 text-sm">
                  <RefreshCw size={15} />
                  Retry search
                </button>
              </div>
            )}

            {!loading && results.length > 0 && (
              <>
                <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="text-sm font-extrabold text-slate-500">{results.length} results</p>
                    <h2 className="font-display text-2xl font-black text-slate-950">{query}</h2>
                  </div>
                </div>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
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
              </>
            )}

            {!loading && searched && results.length === 0 && !error && backendStatus !== 'offline' && (
              <div className="rounded-3xl border border-slate-200 bg-white px-6 py-16 text-center shadow-sm">
                <h3 className="font-display text-xl font-black text-slate-950">No results found</h3>
                <p className="mt-2 text-sm text-slate-500">Try a different query or remove content type filters.</p>
              </div>
            )}
          </section>
        </div>
      </main>

      <SimilarDrawer
        open={drawer.open}
        onClose={() => setDrawer(current => ({ ...current, open: false }))}
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

      <ItemDetailModal item={selectedItem} onClose={() => setSelectedItem(null)} userId={userId} query={query} onSimilar={handleSimilar} onToast={addToast} />
      <Toast toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}