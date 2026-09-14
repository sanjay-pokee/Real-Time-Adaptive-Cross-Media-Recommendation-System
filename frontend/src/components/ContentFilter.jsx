import { motion } from 'framer-motion';
import {
  BookOpen,
  Factory,
  Film,
  HeartPulse,
  LayoutGrid,
  Music,
  Tv,
  Landmark,
} from 'lucide-react';

/**
 * Narrow results to one content type.
 *
 * The list follows the selected domain rather than being fixed: picking Health
 * should not still offer Movies. When no domain is selected every known type is
 * offered. Types come from /domains, so a new one needs only an icon and tint
 * here — and falls back to a neutral pair if even that is missing.
 */
const TYPE_STYLE = {
  movie:      { label: 'Movies',     icon: Film,       tint: 'var(--type-movie)' },
  show:       { label: 'Shows',      icon: Tv,         tint: 'var(--type-show)' },
  book:       { label: 'Books',      icon: BookOpen,   tint: 'var(--type-book)' },
  music:      { label: 'Music',      icon: Music,      tint: 'var(--type-music)' },
  health:     { label: 'Health',     icon: HeartPulse, tint: 'var(--type-health)' },
  industrial: { label: 'Industrial', icon: Factory,    tint: 'var(--type-industrial)' },
  finance:    { label: 'Finance',    icon: Landmark,   tint: 'var(--type-finance)' },
};

const FALLBACK = { icon: LayoutGrid, tint: 'var(--accent)' };

function titleise(name) {
  return String(name).charAt(0).toUpperCase() + String(name).slice(1);
}

export default function ContentFilter({ value, onChange, contentTypes = [] }) {
  const options = [
    { value: null, label: 'All', icon: LayoutGrid, tint: 'var(--accent)' },
    ...contentTypes.map((name) => {
      const style = TYPE_STYLE[name] || FALLBACK;
      return {
        value: name,
        label: style.label || titleise(name),
        icon: style.icon,
        tint: style.tint,
      };
    }),
  ];

  return (
    <div className="grid grid-cols-2 gap-1.5">
      {options.map(({ label, value: filterValue, icon: Icon, tint }) => {
        const active = filterValue === value;
        return (
          <motion.button
            key={String(filterValue)}
            type="button"
            onClick={() => onChange(filterValue)}
            whileTap={{ scale: 0.96 }}
            aria-pressed={active}
            className="relative flex items-center justify-center gap-1.5 overflow-hidden rounded-xl border px-3 py-2 text-xs font-semibold transition-colors"
            style={{
              '--tint': tint,
              borderColor: active ? 'color-mix(in oklab, var(--tint) 55%, transparent)' : 'var(--line)',
              background: active
                ? 'color-mix(in oklab, var(--tint) 15%, transparent)'
                : 'color-mix(in oklab, var(--surface-3) 70%, transparent)',
              color: active ? tint : 'var(--ink-muted)',
              boxShadow: active
                ? '0 0 18px -8px color-mix(in oklab, var(--tint) 80%, transparent)'
                : 'none',
            }}
          >
            <Icon size={13} aria-hidden="true" />
            {label}
          </motion.button>
        );
      })}
    </div>
  );
}
