import { useCallback, useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Loader2, Film, BookOpen, Music, User, Tag } from 'lucide-react';
import { getSuggestions } from '../api/client';

function SuggestionIcon({ kind, hint }) {
  if (kind === 'person') return <User size={14} style={{ color: 'var(--sig-profile)' }} className="shrink-0" />;
  if (kind === 'category') return <Tag size={14} style={{ color: 'var(--warn)' }} className="shrink-0" />;
  // title — derive from hint string e.g. "Movie · Action"
  const h = (hint || '').toLowerCase();
  if (h.startsWith('movie') || h.startsWith('film')) return <Film size={14} style={{ color: 'var(--type-movie)' }} className="shrink-0" />;
  if (h.startsWith('book')) return <BookOpen size={14} style={{ color: 'var(--type-book)' }} className="shrink-0" />;
  if (h.startsWith('music') || h.startsWith('song') || h.startsWith('track')) return <Music size={14} style={{ color: 'var(--type-music)' }} className="shrink-0" />;
  return <Search size={14} className="shrink-0 text-ink-faint" />;
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
        <Search size={18} className="pointer-events-none absolute left-3.5 top-1/2 z-10 -translate-y-1/2 text-ink-faint" />
        {sugLoading && (
          <Loader2
            size={14}
            className="pointer-events-none absolute right-3.5 top-1/2 z-10 -translate-y-1/2 animate-spin text-accent"
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
          className="field h-11 w-full pl-10 pr-9 font-medium"
        />

        {/* Dropdown */}
        <AnimatePresence>
          {open && suggestions.length > 0 && (
            // Animates transform only, never opacity: the dropdown sits over the
            // results grid, so a frame-starved or interrupted animation must not
            // be able to leave it semi-transparent and unreadable.
            <motion.ul
              initial={{ y: -6, scale: 0.985 }}
              animate={{ y: 0, scale: 1 }}
              exit={{ y: -6, scale: 0.985 }}
              transition={{ duration: 0.14, ease: 'easeOut' }}
              className="panel absolute left-0 right-0 top-[calc(100%+6px)] z-50 overflow-hidden shadow-lg"
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
                  className={`flex cursor-pointer items-center gap-2.5 px-3.5 py-2 transition-colors ${
                    idx === activeIdx ? 'bg-accent-soft' : 'hover:bg-surface-3'
                  } ${idx !== 0 ? 'border-t border-line' : ''}`}
                >
                  <SuggestionIcon kind={sug.kind} hint={sug.hint} />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-[13px] font-semibold text-ink">{sug.label}</p>
                    <p className="truncate text-[10.5px] text-ink-faint">{sug.hint}</p>
                  </div>
                  {sug.kind === 'person' && (
                    <span className="chip shrink-0">
                      Person
                    </span>
                  )}
                  {sug.kind === 'title' && (
                    <span className="chip shrink-0">
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
        className="btn btn-primary h-11 px-6 sm:w-auto"
      >
        {loading ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
        <span>{loading ? 'Searching' : 'Search'}</span>
      </motion.button>
    </div>
  );
}