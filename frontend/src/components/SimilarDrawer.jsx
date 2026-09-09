import { AnimatePresence, motion } from 'framer-motion';
import { AlertCircle, GitBranch, Loader2, X } from 'lucide-react';
import RecommendationCard from './RecommendationCard';

export default function SimilarDrawer({
  open,
  onClose,
  title,
  results,
  loading,
  error,
  userId,
  query,
  onSimilar,
  onView,
  onToast,
}) {
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-black/65 backdrop-blur-sm"
          />
          <motion.aside
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', stiffness: 320, damping: 34 }}
            className="fixed bottom-0 right-0 top-0 z-50 w-full max-w-lg overflow-y-auto border-l border-line bg-bg shadow-lg"
            aria-label={`Items similar to ${title}`}
          >
            <div className="sticky top-0 z-10 flex items-center gap-2.5 border-b border-line bg-bg/85 px-4 py-3 backdrop-blur-xl">
              <GitBranch size={14} className="shrink-0 text-accent" />
              <div className="min-w-0 flex-1">
                <p className="label">Similar to</p>
                <p className="display truncate text-[13px] font-bold text-ink">{title}</p>
              </div>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close"
                className="btn btn-ghost h-8 w-8 shrink-0 p-0"
              >
                <X size={14} />
              </button>
            </div>

            <div className="flex flex-col gap-3 p-4">
              {loading && (
                <div className="flex flex-col items-center gap-3 py-20 text-ink-muted">
                  <Loader2 size={26} className="animate-spin text-accent" />
                  <p className="text-[13px]">Finding similar content</p>
                </div>
              )}

              {error && !loading && (
                <div className="flex flex-col items-center gap-2.5 py-16" style={{ color: 'var(--bad)' }}>
                  <AlertCircle size={26} />
                  <p className="text-[13px] font-semibold">Failed to load similar items</p>
                </div>
              )}

              {!loading && !error && results.length === 0 && (
                <div className="flex flex-col items-center gap-2.5 py-16 text-ink-faint">
                  <GitBranch size={26} />
                  <p className="text-[13px]">No similar items found</p>
                </div>
              )}

              {!loading &&
                results.map((result, index) => (
                  <RecommendationCard
                    key={result.global_id}
                    result={result}
                    index={index}
                    userId={userId}
                    query={query}
                    onSimilar={onSimilar}
                    onView={onView}
                    onToast={onToast}
                  />
                ))}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
