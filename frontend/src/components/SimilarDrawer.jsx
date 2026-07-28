import { AnimatePresence, motion } from 'framer-motion';
import { AlertCircle, GitBranch, Loader2, X } from 'lucide-react';
import RecommendationCard from './RecommendationCard';

export default function SimilarDrawer({ open, onClose, title, results, loading, error, userId, query, onSimilar, onView, onToast }) {
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} className="fixed inset-0 z-40 bg-slate-950/45 backdrop-blur-sm" />
          <motion.aside
            initial={{ x: '100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '100%', opacity: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 32 }}
            className="fixed bottom-0 right-0 top-0 z-50 w-full max-w-xl overflow-y-auto border-l border-slate-200 bg-slate-50 shadow-soft"
          >
            <div className="sticky top-0 z-10 flex items-center gap-3 border-b border-slate-200 bg-white/90 px-5 py-4 backdrop-blur-xl">
              <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-blue-50 text-blue-700">
                <GitBranch size={16} />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-xs font-bold uppercase tracking-[0.12em] text-slate-400">Similar to</p>
                <p className="truncate font-display text-sm font-black text-slate-950">{title}</p>
              </div>
              <button type="button" onClick={onClose} className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-500 transition hover:text-slate-950">
                <X size={17} />
              </button>
            </div>

            <div className="flex flex-col gap-4 p-4">
              {loading && <div className="flex flex-col items-center gap-4 py-16 text-slate-500"><Loader2 size={34} className="animate-spin text-blue-600" /><p className="text-sm font-extrabold">Finding similar content</p></div>}
              {error && !loading && <div className="flex flex-col items-center gap-3 py-12 text-red-600"><AlertCircle size={32} /><p className="text-sm font-extrabold">Failed to load similar items</p></div>}
              {!loading && !error && results.length === 0 && <div className="flex flex-col items-center gap-3 py-12 text-slate-400"><GitBranch size={32} /><p className="text-sm font-extrabold">No similar items found</p></div>}
              {!loading && results.map((result, index) => <RecommendationCard key={result.global_id} result={result} index={index} userId={userId} query={query} onSimilar={onSimilar} onView={onView} onToast={onToast} />)}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}