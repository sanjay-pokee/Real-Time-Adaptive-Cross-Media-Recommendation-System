import { AnimatePresence, motion } from 'framer-motion';
import { Info } from 'lucide-react';

/**
 * The legal advisory a regulated domain carries.
 *
 * Health and finance are risk_tier 1 in config/domains.yaml, and the backend
 * returns their advisory text on every response. This renders it verbatim
 * rather than paraphrasing: the wording is the part that matters, and the
 * engine does product discovery, never medical or financial advice.
 *
 * role="status" not "alert" — it is standing context for the results, not an
 * error, so it should not interrupt a screen reader mid-sentence.
 */
export default function AdvisoryBanner({ advisory, tint = 'var(--warn)' }) {
  return (
    <AnimatePresence initial={false}>
      {advisory ? (
        <motion.div
          key={advisory}
          role="status"
          initial={{ opacity: 0, height: 0, marginBottom: 0 }}
          animate={{ opacity: 1, height: 'auto', marginBottom: 14 }}
          exit={{ opacity: 0, height: 0, marginBottom: 0 }}
          transition={{ duration: 0.28, ease: [0.2, 0.8, 0.2, 1] }}
          className="overflow-hidden"
        >
          <div
            className="glass glass-lit flex items-start gap-3 px-4 py-3"
            style={{
              '--tint': tint,
              borderColor: 'color-mix(in oklab, var(--tint) 40%, transparent)',
              background: 'color-mix(in oklab, var(--tint) 9%, var(--glass))',
            }}
          >
            <span
              className="mt-px grid h-6 w-6 shrink-0 place-items-center rounded-full"
              style={{
                background: 'color-mix(in oklab, var(--tint) 20%, transparent)',
                color: 'var(--tint)',
              }}
            >
              <Info size={13} aria-hidden="true" />
            </span>
            <p className="text-[0.8125rem] leading-relaxed" style={{ color: 'var(--ink-muted)' }}>
              {advisory}
            </p>
          </div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
