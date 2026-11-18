import { motion } from 'framer-motion';
import { LayoutGrid } from 'lucide-react';
import { FALLBACK_TYPE_STYLE, TYPE_STYLE, contentTypeLabel } from '../utils/contentTypes';

/**
 * Narrow results to one content type.
 *
 * The list follows the selected domain rather than being fixed: picking Health
 * should not still offer Movies. When no domain is selected every known type is
 * offered. Labels, icons and tints live in utils/contentTypes.js.
 */
export default function ContentFilter({
  value,
  onChange,
  contentTypes = [],
  className = 'grid grid-cols-2 gap-1.5',
}) {
  const options = [
    { value: null, label: 'All', icon: LayoutGrid, tint: 'var(--accent)' },
    ...contentTypes.map((name) => {
      const style = TYPE_STYLE[name] || FALLBACK_TYPE_STYLE;
      return {
        value: name,
        label: contentTypeLabel(name),
        icon: style.icon,
        tint: style.tint,
      };
    }),
  ];

  return (
    <div className={className}>
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
