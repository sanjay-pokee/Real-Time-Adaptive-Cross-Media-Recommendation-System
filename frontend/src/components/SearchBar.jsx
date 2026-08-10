import { useCallback, useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Loader2, Film, BookOpen, Music, User, Tag } from 'lucide-react';
import { getSuggestions } from '../api/client';

const KIND_ICON = {
  title: null,        // will use content_type icon from hint
  person: User,
  category: Tag,
};

function SuggestionIcon({ kind, hint }) {
  if (kind === 'person') return <User size={14} className="shrink-0 text-violet-400" />;
  if (kind === 'category') return <Tag size={14} className="shrink-0 text-amber-400" />;
  // title — derive from hint string e.g. "Movie · Action"
  const h = (hint || '').toLowerCase();
  if (h.startsWith('movie') || h.startsWith('film')) return <Film size={14} className="shrink-0 text-blue-400" />;
  if (h.startsWith('book')) return <BookOpen size={14} className="shrink-0 text-emerald-400" />;
  if (h.startsWith('music') || h.startsWith('song') || h.startsWith('track')) return <Music size={14} className="shrink-0 text-pink-400" />;
  return <Search size={14} className="shrink-0 text-slate-400" />;
}

export default function SearchBar({ value, onChange, onSearch, loading, disabled }) {
  const [suggestions, setSuggestions] = useState([]);
  const [sugLoading, setSugLoading] = useState(false);
  const [activeIdx, setActiveIdx] = useState(-1);
  const [open, setOpen] = useState(false);
  const debounceRef = useRef(null);
  const containerRef = useRef(null);
  const inputRef = useRef(null);

  // ── Fetch suggestions with 250 ms debounce ──────────────────────────────
  const fetchSuggestions = useCallback(async (q) => {
    if (!q || q.trim().length < 2) {
      setSuggestions([]);
      setOpen(false);
      return;
    }
    setSugLoading(true);
    try {
      const data = await getSuggestions(q.trim(), 8);
      setSuggestions(data.suggestions || []);
      setOpen((data.suggestions || []).length > 0);
    } catch {
      setSuggestions([]);
      setOpen(false);
    } finally {
      setSugLoading(false);
    }
  }, []);

  useEffect(() => {
    clearTimeout(debounceRef.current);
    setActiveIdx(-1);
    debounceRef.current = setTimeout(() => fetchSuggestions(value), 250);
    return () => clearTimeout(debounceRef.current);
  }, [value, fetchSuggestions]);

  // ── Close on outside click ───────────────────────────────────────────────
  useEffect(() => {
    function handleClick(e) {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  // ── Keyboard navigation ──────────────────────────────────────────────────
  function handleKey(e) {
    if (open && suggestions.length) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveIdx(i => Math.min(i + 1, suggestions.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveIdx(i => Math.max(i - 1, -1));
        return;
      }
      if (e.key === 'Escape') {
        setOpen(false);
        return;
      }
      if (e.key === 'Enter' && activeIdx >= 0) {
        e.preventDefault();
        selectSuggestion(suggestions[activeIdx]);
        return;
      }
    }
    if (e.key === 'Enter' && !loading && !disabled) {
      setOpen(false);
      onSearch();
    }
  }

  function selectSuggestion(sug) {
    onChange(sug.query);
    setSuggestions([]);
    setOpen(false);
    setActiveIdx(-1);
    onSearch && onSearch(sug.query);
  }

  return (
    <div className="flex flex-col gap-3 sm:flex-row" ref={containerRef}>
      {/* Input wrapper */}
      <div className="relative flex-1">
        <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none z-10" />
        {sugLoading && (
          <Loader2
            size={14}
            className="absolute right-4 top-1/2 -translate-y-1/2 animate-spin text-blue-400 pointer-events-none z-10"
          />
        )}
        <input
          id="search-query"
          ref={inputRef}
          type="text"
          autoComplete="off"
          value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={handleKey}
          onFocus={() => suggestions.length > 0 && setOpen(true)}
          placeholder="Search by title, actor, director, or genre…"
          disabled={disabled}
          className="app-input h-12 pl-11 pr-9 text-sm font-semibold disabled:cursor-not-allowed disabled:bg-slate-100 w-full"
        />

        {/* Dropdown */}
        <AnimatePresence>
          {open && suggestions.length > 0 && (
            <motion.ul
              initial={{ opacity: 0, y: -6, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -6, scale: 0.98 }}
              transition={{ duration: 0.14, ease: 'easeOut' }}
              className="absolute left-0 right-0 top-[calc(100%+6px)] z-50 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-xl"
              role="listbox"
              aria-label="Search suggestions"
            >
              {suggestions.map((sug, idx) => (
                <li
                  key={`${sug.kind}-${sug.label}`}
                  role="option"
                  aria-selected={idx === activeIdx}
                  onMouseDown={e => { e.preventDefault(); selectSuggestion(sug); }}
                  onMouseEnter={() => setActiveIdx(idx)}
                  className={`flex cursor-pointer items-center gap-3 px-4 py-2.5 transition-colors ${
                    idx === activeIdx
                      ? 'bg-blue-50'
                      : 'hover:bg-slate-50'
                  } ${idx !== 0 ? 'border-t border-slate-100' : ''}`}
                >
                  <SuggestionIcon kind={sug.kind} hint={sug.hint} />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-bold text-slate-900">{sug.label}</p>
                    <p className="truncate text-xs font-medium text-slate-400">{sug.hint}</p>
                  </div>
                  {sug.kind === 'person' && (
                    <span className="shrink-0 rounded-full bg-violet-50 px-2 py-0.5 text-[10px] font-extrabold uppercase tracking-wide text-violet-600">
                      Person
                    </span>
                  )}
                  {sug.kind === 'title' && (
                    <span className="shrink-0 rounded-full bg-blue-50 px-2 py-0.5 text-[10px] font-extrabold uppercase tracking-wide text-blue-600">
                      Title
                    </span>
                  )}
                </li>
              ))}
            </motion.ul>
          )}
        </AnimatePresence>
      </div>

      <motion.button
        id="search-btn"
        type="button"
        onClick={() => { setOpen(false); onSearch(); }}
        disabled={loading || disabled || !value.trim()}
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.98 }}
        className="btn-primary flex h-12 items-center justify-center gap-2 px-6 text-sm sm:w-auto"
      >
        {loading ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
        <span>{loading ? 'Searching' : 'Search'}</span>
      </motion.button>
    </div>
  );
}