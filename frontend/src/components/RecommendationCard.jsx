import { motion } from 'framer-motion';
import { ArrowUpRight, BookOpen, Film, Music, Star } from 'lucide-react';
import InteractionButtons from './InteractionButtons';
import ScoreBreakdown from './ScoreBreakdown';
import { coverFor, monogram } from '../utils/cover';

const TYPE_META = {
  movie: { label: 'Movie', icon: Film, color: 'var(--type-movie)' },
  book: { label: 'Book', icon: BookOpen, color: 'var(--type-book)' },
  music: { label: 'Music', icon: Music, color: 'var(--type-music)' },
};

function splitCategories(value, limit = 3) {
  return String(value || '')
    .split(/[,|;]+/)
    .map((entry) => entry.trim())
    .filter(Boolean)
    .slice(0, limit);
}

/** "John Schultz, Carter Jenkins, ..." -> "John Schultz +5" */
function leadCreator(value) {
  const names = String(value || '')
    .split(/[,|;]+/)
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (names.length === 0) return null;
  return { lead: names[0], more: names.length - 1 };
}

export default function RecommendationCard({
  result,
  index,
  userId,
  query,
  onSimilar,
  onView,
  onToast,
}) {
  const meta = TYPE_META[result.content_type] || TYPE_META.movie;
  const Icon = meta.icon;
  const cover = coverFor(result);
  const categories = splitCategories(result.categories);
  const creator = leadCreator(result.creators);
  const year = result.release_date ? String(result.release_date).slice(0, 4) : null;
  const rating = result.rating != null && result.rating !== '' ? Number(result.rating) : null;

  return (
    <motion.article
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index, 8) * 0.035, duration: 0.42, ease: [0.22, 1, 0.36, 1] }}
      className="card group flex flex-col p-4"
    >
      {/* ---------- head: cover + identity ---------- */}
      <div className="flex gap-3.5">
        <button
          type="button"
          onClick={() => onView?.(result)}
          className="relative h-16 w-16 shrink-0 overflow-hidden rounded-xl ring-1 ring-inset ring-white/10 transition-transform duration-300 group-hover:scale-[1.04]"
          style={{ background: cover.background }}
          aria-label={`Open ${result.title}`}
        >
          <span className="absolute inset-0 flex items-center justify-center display text-lg font-extrabold text-white/95 drop-shadow">
            {monogram(result.title)}
          </span>
          <Icon
            size={12}
            className="absolute bottom-1 right-1 text-white/80 drop-shadow"
            strokeWidth={2.5}
          />
        </button>

        <div className="flex min-w-0 flex-1 flex-col">
          <div className="mb-1 flex items-center gap-2">
            <span
              className="text-[10px] font-bold uppercase tracking-[0.09em]"
              style={{ color: meta.color }}
            >
              {meta.label}
            </span>
            {result.source && (
              <span className="truncate text-[10px] font-medium text-ink-faint">
                {result.source}
              </span>
            )}
            <span className="num ml-auto shrink-0 text-[11px] font-semibold tabular-nums text-ink-faint">
              #{index + 1}
            </span>
          </div>

          <button type="button" onClick={() => onView?.(result)} className="text-left">
            <h3 className="display clamp-2 text-[15px] font-bold leading-snug text-ink transition-colors group-hover:text-accent">
              {result.title}
            </h3>
          </button>

          {/* meta line - only renders separators for values that exist */}
          <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-ink-faint">
            {rating != null && (
              <span className="flex items-center gap-1 font-semibold text-warn">
                <Star size={10} fill="currentColor" />
                <span className="num tabular-nums">{rating.toFixed(1)}</span>
              </span>
            )}
            {year && <span className="num tabular-nums">{year}</span>}
            {creator && (
              <span className="min-w-0 truncate">
                {creator.lead}
                {creator.more > 0 && <span className="opacity-60"> +{creator.more}</span>}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* ---------- description ---------- */}
      {result.description ? (
        <button type="button" onClick={() => onView?.(result)} className="mt-3 text-left">
          <p className="clamp-2 text-[12.5px] leading-relaxed text-ink-muted">
            {result.description}
          </p>
        </button>
      ) : (
        <p className="mt-3 text-[12.5px] italic text-ink-faint">No overview available.</p>
      )}

      {/* ---------- categories ---------- */}
      {categories.length > 0 && (
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {categories.map((category) => (
            <span key={category} className="chip">
              {category}
            </span>
          ))}
        </div>
      )}

      {/* ---------- score explanation (pushed to the bottom so cards align) ---------- */}
      <div className="mt-auto pt-4">
        <div className="border-t border-line pt-3.5">
          <ScoreBreakdown result={result} />
        </div>

        <InteractionButtons
          globalId={result.global_id}
          userId={userId}
          query={query}
          onToast={onToast}
          onView={() => onView?.(result)}
        />

        <button
          type="button"
          onClick={() => onSimilar(result)}
          className="btn btn-ghost mt-2.5 w-full py-2"
        >
          Find similar
          <ArrowUpRight size={13} />
        </button>
      </div>
    </motion.article>
  );
}
