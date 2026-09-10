import { motion } from 'framer-motion';
import {
  Factory,
  HeartPulse,
  Landmark,
  Layers,
  ShieldAlert,
  Sparkles,
} from 'lucide-react';

/**
 * Pick which vertical the engine answers from.
 *
 * The options are whatever /domains returns, not a hardcoded list, so a domain
 * pack added to config/domains.yaml appears here on its own. Only the icon and
 * tint are looked up locally, with a neutral fallback for a domain this build
 * has never heard of.
 */
const DOMAIN_STYLE = {
  entertainment: { icon: Sparkles,   tint: 'var(--dom-entertainment)' },
  health:        { icon: HeartPulse, tint: 'var(--dom-health)' },
  industry:      { icon: Factory,    tint: 'var(--dom-industry)' },
  finance:       { icon: Landmark,   tint: 'var(--dom-finance)' },
};

const FALLBACK = { icon: Layers, tint: 'var(--accent)' };

export default function DomainSelector({ domains = [], value, onChange }) {
  const options = [
    { name: null, label: 'All domains', risk_tier: 0 },
    ...domains,
  ];

  return (
    <div
      role="radiogroup"
      aria-label="Recommendation domain"
      className="flex flex-col gap-1.5"
    >
      {options.map((domain) => {
        const { icon: Icon, tint } =
          domain.name === null
            ? { icon: Layers, tint: 'var(--accent)' }
            : DOMAIN_STYLE[domain.name] || FALLBACK;
        const active = domain.name === value;
        const regulated = (domain.risk_tier ?? 0) >= 1;

        return (
          <button
            key={domain.name ?? '__all__'}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(domain.name)}
            title={domain.description || undefined}
            className="group relative flex items-center gap-2.5 rounded-xl px-3 py-2.5 text-left transition-colors"
            style={{
              '--tint': tint,
              color: active ? tint : 'var(--ink-muted)',
            }}
          >
            {/* One shared element slides between rows instead of each row
                fading its own background, so the selection reads as a single
                object moving. layout animations run on transform. */}
            {active ? (
              <motion.span
                layoutId="domain-pill"
                transition={{ type: 'spring', stiffness: 420, damping: 34 }}
                className="absolute inset-0 rounded-xl border"
                style={{
                  borderColor: 'color-mix(in oklab, var(--tint) 45%, transparent)',
                  background: 'color-mix(in oklab, var(--tint) 13%, transparent)',
                  boxShadow: '0 0 22px -6px color-mix(in oklab, var(--tint) 60%, transparent)',
                }}
              />
            ) : null}

            <Icon
              size={15}
              aria-hidden="true"
              className="relative shrink-0 transition-transform group-hover:scale-110"
            />
            <span className="relative min-w-0 flex-1 truncate text-[0.8125rem] font-semibold">
              {domain.label}
            </span>
            {regulated ? (
              <ShieldAlert
                size={12}
                aria-label="Regulated domain — carries an advisory"
                className="relative shrink-0 opacity-70"
              />
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
