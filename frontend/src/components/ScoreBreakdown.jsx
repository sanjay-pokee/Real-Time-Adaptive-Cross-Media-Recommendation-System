import { AnimatePresence, motion } from 'framer-motion';
import { ChevronDown, Info } from 'lucide-react';
import { useState } from 'react';
import { decomposeScore } from '../utils/scoring';

/**
 * Shows how a result's ranking score was actually composed.
 *
 * Rather than plotting each raw signal on its own bar - which implies they
 * matter equally when they do not - this renders one stacked bar where each
 * segment is sized by the points that signal contributed to the final score.
 */
export default function ScoreBreakdown({ result, expandable = true, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  const [hovered, setHovered] = useState(null);

  const { final, parts, inactive } = decomposeScore(result);
  const positiveParts = parts.filter((part) => part.points > 0);

  return (
    <div className="flex flex-col gap-2.5">
      {/* ---- header: label + the number that ranked it ---- */}
      <div className="flex items-baseline justify-between gap-3">
        <span className="label">Ranking score</span>
        <span className="num display text-[15px] font-bold tabular-nums text-ink">
          {final.toFixed(3)}
        </span>
      </div>

      {/* ---- stacked contribution bar ---- */}
      <div
        className="flex h-2 w-full gap-[2px] overflow-hidden rounded-full bg-surface-3 ring-1 ring-inset ring-line"
        role="img"
        aria-label={`Score ${final.toFixed(3)} composed of ${positiveParts
          .map((part) => `${part.label} ${part.points.toFixed(3)}`)
          .join(', ')}`}
      >
        {positiveParts.map((part, index) => (
          <motion.div
            key={part.key}
            initial={{ width: 0 }}
            animate={{ width: `${part.share * 100}%` }}
            transition={{
              duration: 0.55,
              delay: 0.06 * index,
              ease: [0.22, 1, 0.36, 1],
            }}
            onMouseEnter={() => setHovered(part.key)}
            onMouseLeave={() => setHovered(null)}
            className="h-full first:rounded-l-full last:rounded-r-full"
            style={{
              background: part.color,
              opacity: hovered && hovered !== part.key ? 0.3 : 1,
              transition: 'opacity .16s ease',
            }}
            title={`${part.label}: +${part.points.toFixed(3)}`}
          />
        ))}
      </div>

      {/* ---- legend: active signals with the points they added ---- */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        {positiveParts.map((part) => (
          <button
            key={part.key}
            type="button"
            onMouseEnter={() => setHovered(part.key)}
            onMouseLeave={() => setHovered(null)}
            className="group flex items-center gap-1.5 text-left"
            title={part.what}
          >
            <span
              className="h-2 w-2 shrink-0 rounded-full transition-transform group-hover:scale-125"
              style={{ background: part.color }}
            />
            <span className="text-[11px] font-medium text-ink-muted transition-colors group-hover:text-ink">
              {part.label}
            </span>
            <span
              className="num text-[11px] font-semibold tabular-nums"
              style={{ color: part.color }}
            >
              {part.points >= 0 ? '+' : ''}
              {part.points.toFixed(3)}
            </span>
          </button>
        ))}

        {/* Inactive signals stay visible - a dormant signal is information,
            not an error, so we say why it is dormant instead of hiding it. */}
        {inactive.map((signal) => (
          <span
            key={signal.key}
            className="flex cursor-help items-center gap-1.5 opacity-45 transition-opacity hover:opacity-80"
            title={`${signal.label} inactive - ${signal.whyOff}`}
          >
            <span className="h-2 w-2 shrink-0 rounded-full border border-dashed border-ink-faint" />
            <span className="text-[11px] font-medium text-ink-faint line-through decoration-ink-faint/50">
              {signal.label}
            </span>
          </span>
        ))}
      </div>

      {/* ---- expandable arithmetic ---- */}
      {expandable && (
        <>
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            className="flex items-center gap-1.5 self-start text-[11px] font-semibold text-ink-faint transition-colors hover:text-accent"
          >
            <Info size={11} />
            {open ? 'Hide' : 'How this was scored'}
            <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }}>
              <ChevronDown size={11} />
            </motion.span>
          </button>

          <AnimatePresence initial={false}>
            {open && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
                className="overflow-hidden"
              >
                <div className="panel-flat mt-0.5 space-y-1 p-2.5">
                  {parts.map((part) => (
                    <div
                      key={part.key}
                      className="flex items-center justify-between gap-2 text-[11px]"
                    >
                      <span className="flex min-w-0 items-center gap-1.5">
                        <span
                          className="h-1.5 w-1.5 shrink-0 rounded-full"
                          style={{ background: part.color }}
                        />
                        <span className="truncate text-ink-muted">{part.label}</span>
                      </span>
                      <span className="num shrink-0 tabular-nums text-ink-faint">
                        {part.weight !== null && part.raw !== null ? (
                          <>
                            {part.raw.toFixed(3)}
                            <span className="mx-1 opacity-60">x</span>
                            {part.weight.toFixed(2)}
                            <span className="mx-1 opacity-60">=</span>
                          </>
                        ) : (
                          <span className="mr-1 opacity-60">bonus</span>
                        )}
                        <span className="font-semibold text-ink">
                          {part.points >= 0 ? '+' : ''}
                          {part.points.toFixed(3)}
                        </span>
                      </span>
                    </div>
                  ))}

                  <div className="flex items-center justify-between gap-2 border-t border-line pt-1.5 text-[11px]">
                    <span className="font-semibold text-ink-muted">Final</span>
                    <span className="num font-bold tabular-nums text-ink">
                      {final.toFixed(3)}
                    </span>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
    </div>
  );
}
