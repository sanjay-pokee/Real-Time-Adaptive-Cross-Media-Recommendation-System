import { motion } from 'framer-motion';
import { Bookmark, CheckCircle, Eye, Heart, Loader2, SkipForward, Star } from 'lucide-react';
import { useState } from 'react';
import { postInteraction } from '../api/client';
import { useInteractionState } from '../context/InteractionContext';

const ACTIONS = [
  { type: 'view',     icon: Eye,         label: 'View',     active: 'text-blue-700 bg-blue-50 border-blue-200' },
  { type: 'like',     icon: Heart,       label: 'Like',     active: 'text-pink-700 bg-pink-50 border-pink-200' },
  { type: 'bookmark', icon: Bookmark,    label: 'Save',     active: 'text-amber-700 bg-amber-50 border-amber-200' },
  { type: 'skip',     icon: SkipForward, label: 'Skip',     active: 'text-slate-700 bg-slate-100 border-slate-300' },
  { type: 'complete', icon: CheckCircle, label: 'Done',     active: 'text-emerald-700 bg-emerald-50 border-emerald-200' },
];

export default function InteractionButtons({ globalId, userId, query, onToast, onView }) {
  // ── Global synced state (persisted across main feed / drawer / modal) ──
  const {
    active,
    rating,
    toggleActive,
    setRating: setContextRating,
  } = useInteractionState(userId, globalId);

  // ── Local loading spinners only (ephemeral, per-render) ──
  const [loading, setLoading] = useState({});
  const [hoveredStar, setHoveredStar] = useState(0);

  async function handleAction(eventType, eventValue = 1) {
    if (eventType === 'view' && onView) onView();
    if (loading[eventType]) return;
    setLoading(prev => ({ ...prev, [eventType]: true }));
    // Optimistic update — show active state immediately
    toggleActive(eventType);
    try {
      await postInteraction({
        user_id: userId,
        entity_id: globalId,
        event_type: eventType,
        event_value: eventValue,
        context: { source: 'frontend', query: query || '' },
      });
      onToast?.({ type: 'success', message: `${eventType.charAt(0).toUpperCase() + eventType.slice(1)} recorded.` });
    } catch {
      // Revert optimistic toggle on failure
      toggleActive(eventType);
      onToast?.({ type: 'error', title: 'Interaction failed', message: 'Could not save interaction. Is the backend running?' });
    } finally {
      setLoading(prev => ({ ...prev, [eventType]: false }));
    }
  }

  async function handleRating(star) {
    // Update context immediately for snappy UX
    setContextRating(star);
    setLoading(prev => ({ ...prev, rating: true }));
    try {
      await postInteraction({
        user_id: userId,
        entity_id: globalId,
        event_type: 'rating',
        event_value: star,
        context: { source: 'frontend', query: query || '' },
      });
      onToast?.({ type: 'success', message: `Rated ${star}/5 — saved!` });
    } catch {
      onToast?.({ type: 'error', title: 'Rating failed', message: 'Could not save rating. Is the backend running?' });
    } finally {
      setLoading(prev => ({ ...prev, rating: false }));
    }
  }

  return (
    <div className="mt-4 flex flex-col gap-3 border-t border-slate-100 pt-4">
      {/* ── Action buttons ── */}
      <div className="flex flex-wrap items-center gap-1.5">
        {ACTIONS.map(({ type, icon: Icon, label, active: activeClass }) => {
          const isActive = !!active[type];
          const isLoading = !!loading[type];
          return (
            <motion.button
              key={type}
              type="button"
              whileTap={{ scale: 0.96 }}
              onClick={() => handleAction(type)}
              title={label}
              className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-extrabold transition ${isActive ? activeClass : 'border-slate-200 bg-white text-slate-500 hover:bg-slate-50 hover:text-slate-800'}`}
            >
              {isLoading
                ? <Loader2 size={12} className="animate-spin" />
                : <Icon size={12} fill={isActive && type !== 'skip' && type !== 'view' ? 'currentColor' : 'none'} />}
              <span className="hidden sm:inline">{label}</span>
            </motion.button>
          );
        })}
      </div>

      {/* ── Star rating ── */}
      <div className="flex items-center gap-1.5">
        <span className="mr-1 text-xs font-extrabold text-slate-400">Rate</span>
        {[1, 2, 3, 4, 5].map(star => (
          <button
            key={star}
            type="button"
            onMouseEnter={() => setHoveredStar(star)}
            onMouseLeave={() => setHoveredStar(0)}
            onClick={() => handleRating(star)}
            className="text-slate-300 transition hover:text-amber-500"
            disabled={!!loading.rating}
          >
            <Star
              size={15}
              className={star <= (hoveredStar || rating) ? 'text-amber-500' : 'text-slate-300'}
              fill={star <= (hoveredStar || rating) ? 'currentColor' : 'none'}
            />
          </button>
        ))}
        {loading.rating
          ? <Loader2 size={12} className="ml-1 animate-spin text-amber-500" />
          : rating > 0 && <span className="ml-1 text-xs font-black text-amber-600">{rating}/5</span>}
      </div>
    </div>
  );
}