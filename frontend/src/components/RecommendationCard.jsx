import { motion } from 'framer-motion';
import {
  ArrowUpRight,
  BookOpen,
  Factory,
  Film,
  HeartPulse,
  Landmark,
  Music,
  Package,
  Star,
} from 'lucide-react';
import InteractionButtons from './InteractionButtons';
import MaturityBadge from './MaturityBadge';
import ScoreBreakdown from './ScoreBreakdown';
import { coverFor, imageFor, monogram } from '../utils/cover';

const TYPE_META = {
  movie:      { label: 'Movie',      icon: Film,       color: 'var(--type-movie)' },
  book:       { label: 'Book',       icon: BookOpen,   color: 'var(--type-book)' },
  music:      { label: 'Music',      icon: Music,      color: 'var(--type-music)' },
  health:     { label: 'Health',     icon: HeartPulse, color: 'var(--type-health)' },
  industrial: { label: 'Industrial', icon: Factory,    color: 'var(--type-industrial)' },
  finance:    { label: 'Finance',    icon: Landmark,   color: 'var(--type-finance)' },
};

// A content type this build has no styling for should still render, rather
// than silently borrowing the Movie identity.
const FALLBACK_META = { label: 'Item', icon: Package, color: 'var(--accent)' };

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
  const meta = TYPE_META[result.content_type] || FALLBACK_META;
  const Icon = meta.icon;
  const cover = coverFor(result);
  const imageUrl = imageFor(result);
  // Movies carry a 16:9 still; nothing else does. See MEDIA_COLUMNS.
  const backdropUrl = String(result.backdrop_url || '').trim();
  const categories = splitCategories(result.categories);
  const creator = leadCreator(result.creators);
  const year = result.release_date ? String(result.release_date).slice(0, 4) : null;
  const rating = result.rating != null && result.rating !== '' ? Number(result.rating) : null;

  return (
    <motion.article
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index, 8) * 0.035, duration: 0.42, ease: [0.22, 1, 0.36, 1] }}
      className="card group flex flex-col overflow-hidden"
    >
      {/* ---------- media header ----------
          Leads with the picture. Movies carry a landscape still, which fills
          this strip at its own ratio; everything else has a portrait cover,
          which is blurred to fill behind a sharp contained copy rather than
          being stretched into a shape it was never cropped for. The monogram
          gradient sits underneath, so a cover that 404s degrades to the
          gradient instead of an empty box. */}
      <button
        type="button"
        onClick={() => onView?.(result)}
        aria-label={`Open ${result.title}`}
        className="relative block h-36 w-full shrink-0 overflow-hidden text-left"
        style={{ background: cover.background }}
      >
        <span className="absolute inset-0 flex items-center justify-center display text-4xl font-extrabold text-white/20">
          {monogram(result.title)}
        </span>

        {backdropUrl ? (
          <img
            src={backdropUrl}
            alt=""
            loading="lazy"
            decoding="async"
            className="absolute inset-0 h-full w-full object-cover transition-transform duration-[700ms] ease-out group-hover:scale-[1.06]"
            onError={(event) => { event.currentTarget.style.display = 'none'; }}
          />
        ) : imageUrl ? (
          <>
            <img
              src={imageUrl}
              alt=""
              loading="lazy"
              decoding="async"
              className="absolute inset-0 h-full w-full scale-125 object-cover blur-2xl saturate-150"
              onError={(event) => { event.currentTarget.style.display = 'none'; }}
            />
            <img
              src={imageUrl}
              alt=""
              loading="lazy"
              decoding="async"
              className="absolute inset-0 h-full w-full object-contain p-3 drop-shadow-2xl transition-transform duration-[700ms] ease-out group-hover:scale-[1.06]"
              onError={(event) => { event.currentTarget.style.display = 'none'; }}
            />
          </>
        ) : null}

        {/* Scrim: the title sits on an arbitrary photograph, so it needs a
            guaranteed dark base under it rather than luck. */}
        <span className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/30 to-transparent" />

        {/* A specular sweep that only runs on hover - the same language as the
            glass panels, tied to a pointer rather than a timer. */}
        <span className="pointer-events-none absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/12 to-transparent transition-transform duration-[900ms] ease-out group-hover:translate-x-full" />

        <span className="num absolute left-3 top-3 rounded-lg bg-black/55 px-2 py-0.5 text-[11px] font-bold tabular-nums text-white/95 backdrop-blur-sm">
          #{index + 1}
        </span>

        <span
          className="absolute right-3 top-3 flex items-center gap-1.5 rounded-lg bg-black/55 px-2 py-1 text-[10px] font-bold uppercase tracking-[0.09em] backdrop-blur-sm"
          style={{ color: meta.color }}
        >
          <Icon size={11} strokeWidth={2.5} />
          {meta.label}
        </span>

        <span className="absolute inset-x-3 bottom-2.5 block">
          <h3 className="display clamp-2 text-[16px] font-bold leading-snug text-white drop-shadow-[0_2px_6px_rgba(0,0,0,.9)]">
            {result.title}
          </h3>
        </span>
      </button>

      {/* ---------- identity ---------- */}
      {/* flex-1 so the score block below can still be pushed to the bottom with
          mt-auto and every card in a row lines its actions up. */}
      <div className="flex flex-1 flex-col p-4">
        <div className="flex min-w-0 flex-col">
          {result.source && (
            <span className="mb-1 truncate text-[10px] font-medium text-ink-faint">
              {result.source}
            </span>
          )}

          {/* meta line - only renders separators for values that exist */}
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-ink-faint">
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

          {/* The audience rating the eligibility stage filtered on. Showing it
              per item is what makes an age-filtered result set legible: you can
              see that every card really is below the ceiling. */}
          {result.maturity ? (
            <div className="mt-2">
              <MaturityBadge
                maturity={result.maturity}
                minAge={Number(result.audience_min_age)}
              />
            </div>
          ) : null}
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
      </div>
    </motion.article>
  );
}
