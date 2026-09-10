import { motion } from 'framer-motion';
import { ShieldCheck, UserRoundCog } from 'lucide-react';

/**
 * Who the engine is answering for.
 *
 * The three states here mirror the backend policy in backend/audience.py, and
 * the copy is deliberate about the one that surprises people: a request with
 * no age is NOT treated as an adult. It is capped at the teen ceiling, because
 * the eligibility stage fails closed. Showing that on screen turns a policy
 * detail into something a reviewer can see happening.
 */

// Mirrors maturity_levels in config/domains.yaml.
const TIERS = [
  { level: 'restricted', min: 21, label: '21+',      tint: 'var(--mat-restricted)' },
  { level: 'adult',      min: 18, label: '18+',      tint: 'var(--mat-adult)' },
  { level: 'young_adult',min: 16, label: '16+',      tint: 'var(--mat-adult)' },
  { level: 'teen',       min: 13, label: '13+',      tint: 'var(--mat-teen)' },
  { level: 'child',      min: 7,  label: '7+',       tint: 'var(--mat-all_ages)' },
  { level: 'all_ages',   min: 0,  label: 'All ages', tint: 'var(--mat-all_ages)' },
];

const TEEN_CEILING = 13;

function tierFor(age) {
  return TIERS.find((tier) => age >= tier.min) || TIERS[TIERS.length - 1];
}

export default function AudienceControls({
  age,
  onAgeChange,
  safeMode,
  onSafeModeChange,
}) {
  const declared = age !== null && age !== undefined;
  // With no declared age the backend caps at the teen ceiling; with safe mode
  // it caps below a real adult age. Show whichever actually binds.
  const effectiveAge = safeMode ? 7 : declared ? age : TEEN_CEILING;
  const tier = tierFor(effectiveAge);
  const fill = `${Math.round(((declared ? age : TEEN_CEILING) / 80) * 100)}%`;

  return (
    <div className="flex flex-col gap-3.5">
      {/* --- declare an age ------------------------------------------- */}
      <label className="flex cursor-pointer items-center justify-between gap-3">
        <span className="flex items-center gap-2 text-[0.8125rem] font-semibold"
              style={{ color: 'var(--ink)' }}>
          <UserRoundCog size={14} aria-hidden="true" style={{ color: 'var(--accent)' }} />
          Declare viewer age
        </span>
        <input
          type="checkbox"
          className="switch"
          checked={declared}
          onChange={(event) => onAgeChange(event.target.checked ? 18 : null)}
        />
      </label>

      {/* --- the slider ------------------------------------------------ */}
      <div className={declared ? '' : 'pointer-events-none opacity-45'}>
        <div className="mb-1.5 flex items-baseline justify-between">
          <span className="label">Age</span>
          <motion.span
            key={declared ? age : 'none'}
            initial={{ opacity: 0, y: -3 }}
            animate={{ opacity: 1, y: 0 }}
            className="num text-lg font-bold"
            style={{ color: declared ? 'var(--accent)' : 'var(--ink-faint)' }}
          >
            {declared ? age : '—'}
          </motion.span>
        </div>
        <input
          type="range"
          className="slider"
          min={3}
          max={80}
          step={1}
          value={declared ? age : TEEN_CEILING}
          disabled={!declared}
          aria-label="Viewer age"
          style={{ '--fill': fill }}
          onChange={(event) => onAgeChange(Number(event.target.value))}
        />
      </div>

      {/* --- safe mode -------------------------------------------------- */}
      <label className="flex cursor-pointer items-center justify-between gap-3">
        <span className="flex items-center gap-2 text-[0.8125rem] font-semibold"
              style={{ color: 'var(--ink)' }}>
          <ShieldCheck size={14} aria-hidden="true" style={{ color: 'var(--ok)' }} />
          Safe mode
        </span>
        <input
          type="checkbox"
          className="switch"
          checked={safeMode}
          onChange={(event) => onSafeModeChange(event.target.checked)}
        />
      </label>

      {/* --- what the engine will actually enforce ---------------------- */}
      <div
        className="panel-flat flex items-center justify-between gap-2 px-3 py-2.5"
        style={{ borderColor: 'color-mix(in oklab, var(--tint) 35%, transparent)', '--tint': tier.tint }}
      >
        {/* "Ceiling" read as a restriction — a 30-year-old showing "21+"
            looked like they were limited to adult content, when it means the
            opposite. Say which direction it goes. */}
        <span className="label" style={{ letterSpacing: '.06em' }}>Sees up to</span>
        <motion.span
          key={tier.level}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
          className="chip chip-tinted"
          style={{ '--tint': tier.tint }}
        >
          {tier.label}
        </motion.span>
      </div>

      <p className="text-[0.6875rem] leading-relaxed" style={{ color: 'var(--ink-faint)' }}>
        {safeMode
          ? 'Safe mode caps below an adult age, so an adult can ask for family-appropriate results.'
          : declared
            ? 'Ineligible items are excluded in the vector query itself, so they are never scored and cannot be promoted back by any reranker.'
            : 'No age declared. The engine does not assume an adult — it caps at the teen ceiling and fails closed.'}
      </p>
    </div>
  );
}
