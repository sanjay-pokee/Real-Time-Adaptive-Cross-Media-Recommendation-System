import { motion } from 'framer-motion';
import { BookOpen, Calendar, ChevronRight, Film, GitBranch, Music, Star, Tag, TrendingUp, Users } from 'lucide-react';
import InteractionButtons from './InteractionButtons';
import ScoreBars from './ScoreBars';

const TYPE_META = {
  movie: { label: 'Movie', icon: Film, badge: 'badge-movie', color: '#2563EB', bg: '#DBEAFE' },
  book: { label: 'Book', icon: BookOpen, badge: 'badge-book', color: '#059669', bg: '#DCFCE7' },
  music: { label: 'Music', icon: Music, badge: 'badge-music', color: '#DB2777', bg: '#FCE7F3' },
};

export default function RecommendationCard({ result, index, userId, query, onSimilar, onView, onToast }) {
  const meta = TYPE_META[result.content_type] || TYPE_META.movie;
  const Icon = meta.icon;

  return (
    <motion.article
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.04, type: 'spring', stiffness: 260, damping: 24 }}
      className="glass-card group relative flex min-h-[360px] flex-col overflow-hidden p-5"
    >
      <div className="mb-4 flex items-start gap-3">
        <button
          type="button"
          onClick={() => onView?.(result)}
          className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-2xl transition group-hover:scale-105"
          style={{ background: meta.bg, color: meta.color }}
        >
          <Icon size={22} />
        </button>
        <div className="min-w-0 flex-1 pr-8">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span className={meta.badge}>{meta.label}</span>
            {result.source && <span className="truncate text-xs font-bold text-slate-400">{result.source}</span>}
          </div>
          <button type="button" onClick={() => onView?.(result)} className="block text-left">
            <h3 className="line-clamp-2 font-display text-lg font-black leading-tight text-slate-950 transition hover:text-blue-700">{result.title}</h3>
          </button>
        </div>
        <span className="absolute right-5 top-5 flex h-8 w-8 items-center justify-center rounded-full bg-slate-100 text-xs font-black text-slate-500">{index + 1}</span>
      </div>

      {result.description ? (
        <button type="button" onClick={() => onView?.(result)} className="text-left">
          <p className="line-clamp-3 text-sm leading-6 text-slate-600 transition group-hover:text-slate-800">{result.description}</p>
          <span className="mt-2 inline-flex items-center gap-1 text-xs font-extrabold text-blue-700">Open details <ChevronRight size={13} /></span>
        </button>
      ) : (
        <p className="text-sm italic text-slate-400">No overview text available.</p>
      )}

      <div className="mt-4 flex flex-wrap gap-3">
        {result.creators && (
          <div className="flex min-w-0 items-center gap-1.5 text-xs font-semibold text-slate-500">
            <Users size={12} />
            <span className="max-w-[150px] truncate">{result.creators}</span>
          </div>
        )}
        {result.release_date && (
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500">
            <Calendar size={12} />
            <span>{String(result.release_date).slice(0, 7)}</span>
          </div>
        )}
        {result.rating != null && result.rating !== '' && (
          <div className="flex items-center gap-1.5 text-xs font-black text-amber-600">
            <Star size={12} fill="currentColor" />
            <span>{Number(result.rating).toFixed(1)}</span>
          </div>
        )}
        {result.popularity != null && result.popularity !== '' && (
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500">
            <TrendingUp size={12} />
            <span>{Math.round(Number(result.popularity)).toLocaleString()}</span>
          </div>
        )}
      </div>

      {result.categories && (
        <div className="mt-3 flex flex-wrap items-start gap-1.5">
          <Tag size={12} className="mt-1 text-slate-400" />
          {String(result.categories)
            .split(/[,|;]+/)
            .slice(0, 4)
            .map(category => category.trim())
            .filter(Boolean)
            .map(category => <span key={category} className="rounded-full bg-slate-100 px-2 py-1 text-[11px] font-bold text-slate-500">{category}</span>)}
        </div>
      )}

      <div className="mt-auto">
        <ScoreBars score={result.score} semantic_score={result.semantic_score} graph_score={result.graph_score} ema_score={result.ema_score} />
        <InteractionButtons globalId={result.global_id} userId={userId} query={query} onToast={onToast} onView={() => onView?.(result)} />
        <button
          type="button"
          onClick={() => onSimilar(result)}
          className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs font-black text-slate-600 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700"
        >
          <GitBranch size={14} />
          Find similar
          <ChevronRight size={14} />
        </button>
      </div>
    </motion.article>
  );
}