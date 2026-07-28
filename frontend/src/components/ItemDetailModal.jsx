import { AnimatePresence, motion } from 'framer-motion';
import { BookOpen, Calendar, CheckCircle, Film, GitBranch, Music, Sparkles, Star, Tag, TrendingUp, Users, X } from 'lucide-react';
import InteractionButtons from './InteractionButtons';
import ScoreBars from './ScoreBars';

const TYPE_META = {
  movie: { label: 'Movie', icon: Film, badge: 'badge-movie', color: '#2563EB', bg: '#DBEAFE' },
  book: { label: 'Book', icon: BookOpen, badge: 'badge-book', color: '#059669', bg: '#DCFCE7' },
  music: { label: 'Music', icon: Music, badge: 'badge-music', color: '#DB2777', bg: '#FCE7F3' },
};

export default function ItemDetailModal({ item, onClose, userId, query, onSimilar, onToast }) {
  if (!item) return null;

  const meta = TYPE_META[item.content_type] || TYPE_META.movie;
  const Icon = meta.icon;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[99990] flex items-center justify-center overflow-y-auto p-4 sm:p-6">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} className="fixed inset-0 bg-slate-950/55 backdrop-blur-sm" />

        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 18 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: 18 }}
          transition={{ type: 'spring', stiffness: 350, damping: 30 }}
          className="relative z-[99999] my-auto flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-soft"
        >
          <div className="absolute left-0 right-0 top-0 h-1" style={{ background: meta.color }} />
          <button type="button" onClick={onClose} className="absolute right-5 top-5 z-10 flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-500 shadow-sm transition hover:text-slate-950">
            <X size={18} />
          </button>

          <div className="overflow-y-auto p-6 pr-5 sm:p-8 sm:pr-7">
            <div className="flex items-start gap-4 pr-12">
              <div className="flex h-14 w-14 flex-shrink-0 items-center justify-center rounded-2xl" style={{ background: meta.bg, color: meta.color }}>
                <Icon size={27} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <span className={meta.badge}>{meta.label}</span>
                  {item.source && <span className="rounded-lg bg-slate-100 px-2 py-1 text-xs font-bold text-slate-500">{item.source}</span>}
                </div>
                <h2 className="font-display text-2xl font-black leading-tight text-slate-950 sm:text-3xl">{item.title}</h2>
              </div>
            </div>

            <section className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <h4 className="mb-2 flex items-center gap-2 text-xs font-black uppercase tracking-[0.12em] text-slate-500"><Sparkles size={13} className="text-blue-600" />Overview</h4>
              {item.description ? <p className="whitespace-pre-line text-sm leading-7 text-slate-700">{item.description}</p> : <p className="text-sm italic text-slate-400">No extended description available.</p>}
            </section>

            <section className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {item.creators && <MetaTile icon={Users} label="Creator" value={item.creators} />}
              {item.release_date && <MetaTile icon={Calendar} label="Release" value={String(item.release_date)} />}
              {item.rating != null && item.rating !== '' && <MetaTile icon={Star} label="Rating" value={`${Number(item.rating).toFixed(1)} / 10`} tone="amber" />}
              {item.popularity != null && item.popularity !== '' && <MetaTile icon={TrendingUp} label="Popularity" value={Math.round(Number(item.popularity)).toLocaleString()} />}
            </section>

            {item.categories && (
              <section className="mt-6">
                <h4 className="mb-2 flex items-center gap-2 text-xs font-black uppercase tracking-[0.12em] text-slate-500"><Tag size={13} />Categories</h4>
                <div className="flex flex-wrap gap-2">
                  {String(item.categories).split(/[,|;]+/).map(category => category.trim()).filter(Boolean).map(category => <span key={category} className="rounded-full bg-slate-100 px-3 py-1.5 text-xs font-bold text-slate-600">{category}</span>)}
                </div>
              </section>
            )}

            <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-4">
              <h4 className="mb-1 text-xs font-black uppercase tracking-[0.12em] text-slate-500">Recommendation signals</h4>
              <ScoreBars score={item.score} semantic_score={item.semantic_score} graph_score={item.graph_score} ema_score={item.ema_score} />
            </section>

            <section className="mt-6">
              <h4 className="mb-2 flex items-center gap-2 text-xs font-black uppercase tracking-[0.12em] text-slate-500"><CheckCircle size={13} className="text-emerald-600" />Log interaction</h4>
              <InteractionButtons globalId={item.global_id} userId={userId} query={query} onToast={onToast} />
            </section>

            <div className="mt-6 flex flex-col gap-3 border-t border-slate-100 pt-5 sm:flex-row sm:items-center">
              <span className="truncate rounded-xl bg-slate-100 px-3 py-2 text-[11px] font-bold text-slate-500">ID: {item.global_id}</span>
              <button type="button" onClick={() => { onClose(); onSimilar(item); }} className="btn-primary flex w-full items-center justify-center gap-2 px-5 py-2.5 text-xs sm:ml-auto sm:w-auto">
                <GitBranch size={14} />
                Find similar content
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}

function MetaTile({ icon: Icon, label, value, tone = 'slate' }) {
  const toneClass = tone === 'amber' ? 'text-amber-600' : 'text-blue-600';
  return (
    <div className="min-w-0 rounded-2xl border border-slate-200 bg-white p-3 shadow-sm">
      <span className="mb-1 block text-[11px] font-bold text-slate-400">{label}</span>
      <div className={`flex min-w-0 items-center gap-1.5 text-xs font-black ${toneClass}`}>
        <Icon size={13} className="flex-shrink-0" />
        <span className="truncate text-slate-800">{value}</span>
      </div>
    </div>
  );
}