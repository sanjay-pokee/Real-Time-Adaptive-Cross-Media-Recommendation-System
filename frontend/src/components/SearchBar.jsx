import { motion } from 'framer-motion';
import { Search, Loader2 } from 'lucide-react';

export default function SearchBar({ value, onChange, onSearch, loading, disabled }) {
  function handleKey(event) {
    if (event.key === 'Enter' && !loading && !disabled) onSearch();
  }

  return (
    <div className="flex flex-col gap-3 sm:flex-row">
      <div className="relative flex-1">
        <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
        <input
          id="search-query"
          type="text"
          value={value}
          onChange={event => onChange(event.target.value)}
          onKeyDown={handleKey}
          placeholder="Describe a mood, story, topic, or sound"
          disabled={disabled}
          className="app-input h-12 pl-11 pr-4 text-sm font-semibold disabled:cursor-not-allowed disabled:bg-slate-100"
        />
      </div>

      <motion.button
        id="search-btn"
        type="button"
        onClick={onSearch}
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