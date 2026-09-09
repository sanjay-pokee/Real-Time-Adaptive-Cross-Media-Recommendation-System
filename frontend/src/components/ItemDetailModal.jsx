import { AnimatePresence, motion } from 'framer-motion';
import {
  ArrowUpRight,
  BookOpen,
  Calendar,
  Film,
  Music,
  Star,
  TrendingUp,
  Users,
  X,
} from 'lucide-react';
import { useEffect } from 'react';
import InteractionButtons from './InteractionButtons';
import ScoreBreakdown from './ScoreBreakdown';
import { coverFor, monogram } from '../utils/cover';
import { explainRank } from '../utils/scoring';

const TYPE_META = {
  movie: { label: 'Movie', icon: Film, color: 'var(--type-movie)' },
  book: { label: 'Book', icon: BookOpen, color: 'var(--type-book)' },
  music: { label: 'Music', icon: Music, color: 'var(--type-music)' },
};

export default function ItemDetailModal({ item, onClose, userId, query, onSimilar, onToast }) {
  // Close on Escape - registered unconditionally so hook order stays stable.
  useEffect(() => {
    if (!item) return undefined;
    const onKey = (event) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [item, onClose]);

  if (!item) return null;

  const meta = TYPE_META[item.content_type] || TYPE_META.movie;
  const Icon = meta.icon;
  const cover = coverFor(item);
  const categories = String(item.categories || '')
    .split(/[,|;]+/)
    .map((entry) => entry.trim())
    .filter(Boolean);

  return (
    <AnimatePresence>
      <div
        className="fixed inset-0 z-[99990] flex items-center justify-center overflow-y-auto p-4 sm:p-6"
        role="dialog"
        aria-modal="true"
        aria-label={item.title}
      >
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="fixed inset-0 bg-black/70 backdrop-blur-sm"
        />

        <motion.div
          initial={{ opacity: 0, scale: 0.97, y: 14 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.97, y: 14 }}
          transition={{ type: 'spring', stiffness: 360, damping: 32 }}
          className="panel relative z-[99999] my-auto flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden shadow-lg"
        >
          {/* ---- banner ---- */}
          <div className="relative h-24 shrink-0" style={{ background: cover.background }}>
            <div className="absolute inset-0 bg-gradient-to-t from-[var(--surface)] via-transparent to-transparent" />
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="absolute right-3 top-3 flex h-8 w-8 items-center justify-center rounded-lg bg-black/35 text-white/90 backdrop-blur transition-colors hover:bg-black/55"
            >
              <X size={15} />
            </button>
          </div>

          <div className="-mt-9 overflow-y-auto px-6 pb-6 sm:px-7">
            {/* ---- identity ---- */}
            <div className="flex items-end gap-3.5">
              <div
                className="relative flex h-[72px] w-[72px] shrink-0 items-center justify-center rounded-2xl"
                style={{
                  background: cover.background,
                  boxShadow: '0 0 0 4px var(--surface)',
                }}
              >
                <span className="display text-xl font-extrabold text-white/95 drop-shadow">
                  {monogram(item.title)}
                </span>
                <Icon size={13} className="absolute bottom-1.5 right-1.5 text-white/80" strokeWidth={2.5} />
              </div>

              <div className="min-w-0 flex-1 pb-1">
                <div className="mb-1 flex items-center gap-2">
                  <span
                    className="text-[10px] font-bold uppercase tracking-[0.09em]"
                    style={{ color: meta.color }}
                  >
                    {meta.label}
                  </span>
                  {item.source && (
                    <span className="truncate text-[10px] text-ink-faint">{item.source}</span>
                  )}
                </div>
                <h2 className="display text-xl font-bold leading-tight text-ink sm:text-2xl">
                  {item.title}
                </h2>
              </div>
            </div>

            {/* ---- why it ranked ---- */}
            <p className="mt-4 rounded-xl border border-line bg-accent-soft px-3.5 py-2.5 text-[12px] leading-relaxed text-ink">
              {explainRank(item)}
            </p>

            {/* ---- overview ---- */}
            <section className="mt-5">
              <h4 className="label mb-2">Overview</h4>
              {item.description ? (
                <p className="whitespace-pre-line text-[13px] leading-relaxed text-ink-muted">
                  {item.description}
                </p>
              ) : (
                <p className="text-[13px] italic text-ink-faint">No description available.</p>
              )}
            </section>

            {/* ---- facts ---- */}
            <section className="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {item.creators && <MetaTile icon={Users} label="Creators" value={item.creators} />}
              {item.release_date && (
                <MetaTile icon={Calendar} label="Released" value={String(item.release_date)} />
              )}
              {item.rating != null && item.rating !== '' && (
                <MetaTile
                  icon={Star}
                  label="Rating"
                  value={`${Number(item.rating).toFixed(1)} / 10`}
                  tint="var(--warn)"
                />
              )}
              {item.popularity != null && item.popularity !== '' && (
                <MetaTile
                  icon={TrendingUp}
                  label="Popularity"
                  value={Math.round(Number(item.popularity)).toLocaleString()}
                />
              )}
            </section>

            {/* ---- categories ---- */}
            {categories.length > 0 && (
              <section className="mt-5">
                <h4 className="label mb-2">Categories</h4>
                <div className="flex flex-wrap gap-1.5">
                  {categories.map((category) => (
                    <span key={category} className="chip">
                      {category}
                    </span>
                  ))}
                </div>
              </section>
            )}

            {/* ---- score ---- */}
            <section className="panel-flat mt-5 p-4">
              <ScoreBreakdown result={item} defaultOpen />
            </section>

            {/* ---- interactions ---- */}
            <section className="mt-4">
              <h4 className="label">Log interaction</h4>
              <InteractionButtons
                globalId={item.global_id}
                userId={userId}
                query={query}
                onToast={onToast}
              />
            </section>

            <div className="mt-5 flex flex-col gap-2 border-t border-line pt-4 sm:flex-row sm:items-center">
              <span className="num truncate font-mono text-[10px] text-ink-faint">
                {item.global_id}
              </span>
              <button
                type="button"
                onClick={() => {
                  onClose();
                  onSimilar(item);
                }}
                className="btn btn-primary px-4 py-2 sm:ml-auto"
              >
                Find similar
                <ArrowUpRight size={13} />
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}

function MetaTile({ icon: Icon, label, value, tint = 'var(--ink-muted)' }) {
  return (
    <div className="panel-flat min-w-0 px-3 py-2">
      <span className="mb-1 block text-[10px] text-ink-faint">{label}</span>
      <div className="flex min-w-0 items-center gap-1.5">
        <Icon size={11} className="shrink-0" style={{ color: tint }} />
        <span className="truncate text-[11.5px] font-medium text-ink">{value}</span>
      </div>
    </div>
  );
}
