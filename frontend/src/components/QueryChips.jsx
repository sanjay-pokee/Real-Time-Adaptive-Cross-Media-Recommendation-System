import { motion } from 'framer-motion';

/**
 * One-click demonstrations, not just query text.
 *
 * A chip carries the domain and the viewer age it needs, because the
 * interesting queries only make sense with both. "Joint pain supplements" as
 * an undeclared-age viewer returns nothing at all — health is adult-only and
 * an undeclared age caps at the teen ceiling — so a chip that set the query
 * alone would look broken. Setting all three makes each chip a complete,
 * self-explanatory scenario.
 *
 * `age: null` is meaningful and deliberate on the last two: it exercises the
 * undeclared-age path rather than being an omission.
 */
const CHIPS = [
  // --- entertainment -------------------------------------------------
  { tag: 'Sci-fi',   label: 'space adventure with aliens', domain: null,      age: null },
  { tag: 'Kids',     label: 'fun adventure',               domain: null,      age: 8    },
  { tag: 'Contrast', label: 'dark violent thriller',       domain: null,      age: 8    },
  { tag: 'Adult',    label: 'dark violent thriller',       domain: null,      age: 25   },
  // --- the other three verticals -------------------------------------
  { tag: 'Health',   label: 'vitamins for joint pain relief',  domain: 'health',   age: 30 },
  { tag: 'Health',   label: 'blood pressure monitor',          domain: 'health',   age: 30 },
  { tag: 'Industry', label: 'precision measurement caliper',   domain: 'industry', age: 30 },
  { tag: 'Finance',  label: 'tax filing software',             domain: 'finance',  age: 30 },
  { tag: 'Finance',  label: 'personal budget and expenses',    domain: 'finance',  age: 30 },
];

const TAG_TINT = {
  'Sci-fi':   'var(--dom-entertainment)',
  Kids:       'var(--mat-all_ages)',
  Contrast:   'var(--mat-adult)',
  Adult:      'var(--mat-adult)',
  Health:     'var(--dom-health)',
  Industry:   'var(--dom-industry)',
  Finance:    'var(--dom-finance)',
};

export default function QueryChips({ onSelect }) {
  return (
    <div className="flex flex-col gap-1">
      {CHIPS.map((chip, index) => (
        <motion.button
          key={`${chip.tag}-${chip.label}`}
          type="button"
          initial={{ opacity: 0, x: -6 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.03, duration: 0.3 }}
          onClick={() => onSelect(chip)}
          title={
            chip.age === null
              ? 'No age declared — capped at the teen ceiling'
              : `As a ${chip.age}-year-old`
          }
          className="group flex items-center gap-2.5 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-surface-3"
          style={{ '--tint': TAG_TINT[chip.tag] || 'var(--accent)' }}
        >
          <span
            className="w-16 shrink-0 text-[10px] font-bold uppercase tracking-[0.06em] transition-colors"
            style={{ color: 'var(--ink-faint)' }}
          >
            <span className="group-hover:hidden">{chip.tag}</span>
            <span className="hidden group-hover:inline" style={{ color: 'var(--tint)' }}>
              {chip.age === null ? 'no age' : `age ${chip.age}`}
            </span>
          </span>
          <span className="min-w-0 flex-1 truncate text-xs text-ink-muted transition-colors group-hover:text-ink">
            {chip.label}
          </span>
        </motion.button>
      ))}
    </div>
  );
}
