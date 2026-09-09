import { AnimatePresence, motion } from 'framer-motion';
import { useState } from 'react';
import { Activity, Brain, ChevronDown, Layers, Network, Share2, UserRoundCog } from 'lucide-react';
import { SIGNALS } from '../utils/scoring';

const ICONS = {
  semantic: Brain,
  graph: Network,
  ema: Activity,
  kg: Share2,
};

/**
 * Explains the ranking pipeline and shows, per signal, whether it is
 * contributing right now. During a demo this is the panel that answers
 * "what is actually running underneath?".
 */
export default function AISignalsPanel({ topResult }) {
  const [open, setOpen] = useState(false);

  const rows = SIGNALS.map((signal) => {
    const raw = topResult?.[signal.field];
    const value = raw === null || raw === undefined || raw === '' ? null : Number(raw);
    const live = value !== null && Number.isFinite(value) && value !== 0;
    return { ...signal, value, live, Icon: ICONS[signal.key] || Brain };
  });

  const liveCount = rows.filter((row) => row.live).length;

  return (
    <div className="panel overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2.5 px-4 py-3 text-left transition-colors hover:bg-surface-3"
      >
        <Layers size={14} className="text-accent" />
        <span className="flex-1 text-[13px] font-semibold text-ink">Ranking pipeline</span>
        {topResult && (
          <span className="num text-[10px] font-semibold tabular-nums text-ink-faint">
            {liveCount}/{rows.length} live
          </span>
        )}
        <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronDown size={14} className="text-ink-faint" />
        </motion.span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
            className="overflow-hidden"
          >
            <div className="space-y-3 border-t border-line px-4 py-3.5">
              <p className="text-[11px] leading-relaxed text-ink-muted">
                Sentence-BERT retrieves candidates from Qdrant, then each signal
                below adds weighted points to produce the final ranking score.
              </p>

              {rows.map(({ key, label, color, weight, what, whyOff, value, live, Icon }) => (
                <div key={key} className="flex gap-2.5">
                  <div
                    className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-lg"
                    style={{
                      background: live ? `color-mix(in srgb, ${color} 16%, transparent)` : 'var(--surface-3)',
                      color: live ? color : 'var(--ink-faint)',
                      opacity: live ? 1 : 0.6,
                    }}
                  >
                    <Icon size={12} />
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span
                        className="text-[11px] font-semibold"
                        style={{ color: live ? color : 'var(--ink-faint)' }}
                      >
                        {label}
                      </span>
                      <span className="num text-[10px] tabular-nums text-ink-faint">
                        x{weight.toFixed(2)}
                      </span>
                      {live ? (
                        <span className="num ml-auto text-[10px] font-semibold tabular-nums text-ink">
                          {value.toFixed(3)}
                        </span>
                      ) : (
                        <span className="ml-auto text-[10px] text-ink-faint">idle</span>
                      )}
                    </div>
                    <p className="mt-0.5 text-[10.5px] leading-relaxed text-ink-faint">
                      {live ? what : whyOff}
                    </p>
                  </div>
                </div>
              ))}

              <div className="flex gap-2.5 border-t border-line pt-3">
                <div
                  className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-lg"
                  style={{
                    background: 'color-mix(in srgb, var(--sig-profile) 16%, transparent)',
                    color: 'var(--sig-profile)',
                  }}
                >
                  <UserRoundCog size={12} />
                </div>
                <div className="min-w-0 flex-1">
                  <span className="text-[11px] font-semibold text-sig-profile">Profile re-rank</span>
                  <p className="mt-0.5 text-[10.5px] leading-relaxed text-ink-faint">
                    Category and content-type affinity learned from your saved interactions.
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
