import { motion } from 'framer-motion';
import { BookOpen, Film, LayoutGrid, Music } from 'lucide-react';

const FILTERS = [
  { label: 'All', value: null, icon: LayoutGrid, tint: 'var(--accent)' },
  { label: 'Movies', value: 'movie', icon: Film, tint: 'var(--type-movie)' },
  { label: 'Books', value: 'book', icon: BookOpen, tint: 'var(--type-book)' },
  { label: 'Music', value: 'music', icon: Music, tint: 'var(--type-music)' },
];

export default function ContentFilter({ value, onChange }) {
  return (
    <div className="grid grid-cols-2 gap-1.5">
      {FILTERS.map(({ label, value: filterValue, icon: Icon, tint }) => {
        const active = filterValue === value;
        return (
          <motion.button
            key={String(filterValue)}
            type="button"
            onClick={() => onChange(filterValue)}
            whileTap={{ scale: 0.97 }}
            aria-pressed={active}
            className="flex items-center justify-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-semibold transition-colors"
            style={{
              borderColor: active ? tint : 'var(--line)',
              background: active
                ? `color-mix(in srgb, ${tint} 14%, transparent)`
                : 'var(--surface-3)',
              color: active ? tint : 'var(--ink-muted)',
            }}
          >
            <Icon size={13} />
            {label}
          </motion.button>
        );
      })}
    </div>
  );
}
