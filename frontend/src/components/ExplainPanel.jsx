import { motion } from 'framer-motion';
import { ArrowDown, ArrowUp, Minus, ShieldCheck, Sparkles } from 'lucide-react';
import {
  counterfactualRanks,
  decomposeScore,
  explainEligibility,
  explainRank,
} from '../utils/scoring';

/**
 * Why this item ranked here, and what would change without each signal.
 *
 * The counterfactual column is the point of this panel. A stacked
 * contribution bar shows how many points a signal added, but not whether that
 * mattered: a signal can contribute heavily and move nothing, because every
 * competing item gained a similar amount. Re-ranking with the signal removed
 * answers the question directly, and since the fusion is a weighted sum it is
 * exact rather than an approximation.
 */
export default function ExplainPanel({ item, results, effectiveAge }) {
  if (!item) return null;

  const { parts, final } = decomposeScore(item);
  const counterfactuals = counterfactualRanks(results, item);
  const eligibility = explainEligibility(item, effectiveAge);

  return (
    <div className="flex flex-col gap-4">
      {/* ---- prose summary ---- */}
      <p className="flex items-start gap-2 text-[12.5px] leading-relaxed" style={{ color: 'var(--ink)' }}>
        <Sparkles size={13} className="mt-0.5 shrink-0" style={{ color: 'var(--accent)' }} />
        <span>{explainRank(item)}</span>
      </p>

      {/* ---- eligibility ---- */}
      {eligibility ? (
        <div
          className="panel-flat flex items-start gap-2 px-3 py-2.5"
          style={{ borderColor: 'color-mix(in oklab, var(--ok) 30%, transparent)' }}
        >
          <ShieldCheck size={13} className="mt-0.5 shrink-0" style={{ color: 'var(--ok)' }} />
          <div className="min-w-0">
            <p className="text-[11px] font-semibold" style={{ color: 'var(--ink)' }}>
              Passed the eligibility stage
            </p>
            <p className="mt-0.5 text-[10.5px] leading-relaxed" style={{ color: 'var(--ink-muted)' }}>
              {eligibility}
            </p>
          </div>
        </div>
      ) : null}

      {/* ---- contribution bar ---- */}
      <div>
        <div className="mb-1.5 flex items-baseline justify-between">
          <span className="label">Score composition</span>
          <span className="num text-[13px] font-bold" style={{ color: 'var(--ink)' }}>
            {final.toFixed(3)}
          </span>
        </div>
        <div className="track flex h-2" style={{ height: 8 }}>
          {parts
            .filter((part) => part.share > 0)
            .map((part) => (
              <motion.span
                key={part.key}
                initial={{ width: 0 }}
                animate={{ width: `${part.share * 100}%` }}
                transition={{ duration: 0.5, ease: [0.2, 0.8, 0.2, 1] }}
                title={`${part.label}: ${part.points >= 0 ? '+' : ''}${part.points.toFixed(3)}`}
                style={{ background: part.color }}
              />
            ))}
        </div>
        <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1">
          {parts.map((part) => (
            <div key={part.key} className="flex items-center gap-1.5 text-[10.5px]">
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ background: part.color }}
                aria-hidden="true"
              />
              <dt className="min-w-0 flex-1 truncate" style={{ color: 'var(--ink-muted)' }}>
                {part.label}
              </dt>
              <dd className="num shrink-0 font-semibold" style={{ color: 'var(--ink)' }}>
                {part.points >= 0 ? '+' : ''}{part.points.toFixed(3)}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      {/* ---- counterfactuals ---- */}
      {counterfactuals.length ? (
        <div>
          <span className="label">If a signal were removed</span>
          <p className="mb-2 mt-1 text-[10.5px] leading-relaxed" style={{ color: 'var(--ink-faint)' }}>
            Re-ranked with that signal subtracted from every result. A signal can
            add points and still change nothing.
          </p>
          <ul className="flex flex-col gap-1">
            {counterfactuals.map((row) => {
              const moved = row.delta !== 0;
              const Icon = !moved ? Minus : row.delta > 0 ? ArrowDown : ArrowUp;
              // delta > 0 means it would rank WORSE without the signal, i.e.
              // the signal was helping it.
              const tint = !moved
                ? 'var(--ink-faint)'
                : row.delta > 0 ? 'var(--ok)' : 'var(--bad)';
              return (
                <li
                  key={row.key}
                  className="panel-flat flex items-center gap-2 px-2.5 py-1.5 text-[11px]"
                >
                  <span
                    className="h-2 w-2 shrink-0 rounded-full"
                    style={{ background: row.color }}
                    aria-hidden="true"
                  />
                  <span className="min-w-0 flex-1 truncate" style={{ color: 'var(--ink-muted)' }}>
                    without {row.label}
                  </span>
                  <span className="num shrink-0 font-semibold tabular-nums" style={{ color: 'var(--ink-faint)' }}>
                    #{row.actualRank}
                  </span>
                  <Icon size={11} style={{ color: tint }} aria-hidden="true" />
                  <span
                    className="num w-7 shrink-0 text-right font-bold tabular-nums"
                    style={{ color: tint }}
                  >
                    #{row.counterfactualRank}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
