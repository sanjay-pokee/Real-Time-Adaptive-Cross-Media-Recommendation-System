import { motion } from 'framer-motion';
import { Bookmark, Check, Eye, Heart, Loader2, SkipForward, Star } from 'lucide-react';
import { useState } from 'react';
import { postInteraction } from '../api/client';
import { useInteractionState } from '../context/InteractionContext';

const ACTIONS = [
  { type: 'view', icon: Eye, label: 'Mark viewed', tint: 'var(--accent)' },
  { type: 'like', icon: Heart, label: 'Like', tint: 'var(--sig-profile)' },
  { type: 'bookmark', icon: Bookmark, label: 'Save', tint: 'var(--sig-ema)' },
  { type: 'skip', icon: SkipForward, label: 'Not interested', tint: 'var(--ink-muted)' },
  { type: 'complete', icon: Check, label: 'Done', tint: 'var(--sig-graph)' },
];

const FILLED = new Set(['like', 'bookmark']);

export default function InteractionButtons({ globalId, userId, query, onToast, onView }) {
  const { active, rating, toggleActive, setRating: setContextRating } =
    useInteractionState(userId, globalId);

  const [loading, setLoading] = useState({});
  const [hoveredStar, setHoveredStar] = useState(0);

  async function handleAction(eventType, eventValue = 1) {
    if (eventType === 'view' && onView) onView();
    if (loading[eventType]) return;
    setLoading((prev) => ({ ...prev, [eventType]: true }));
    toggleActive(eventType);
    try {
      await postInteraction({
        user_id: userId,
        entity_id: globalId,
        event_type: eventType,
        event_value: eventValue,
        context: { source: 'frontend', query: query || '' },
      });
      onToast?.({
        type: 'success',
        message: `${eventType.charAt(0).toUpperCase() + eventType.slice(1)} recorded.`,
      });
    } catch {
      toggleActive(eventType);
      onToast?.({
        type: 'error',
        title: 'Interaction failed',
        message: 'Could not save interaction. Is the backend running?',
      });
    } finally {
      setLoading((prev) => ({ ...prev, [eventType]: false }));
    }
  }

  async function handleRating(star) {
    setContextRating(star);
    setLoading((prev) => ({ ...prev, rating: true }));
    try {
      await postInteraction({
        user_id: userId,
        entity_id: globalId,
        event_type: 'rating',
        event_value: star,
        context: { source: 'frontend', query: query || '' },
      });
      onToast?.({ type: 'success', message: `Rated ${star}/5 - saved.` });
    } catch {
      onToast?.({
        type: 'error',
        title: 'Rating failed',
        message: 'Could not save rating. Is the backend running?',
      });
    } finally {
      setLoading((prev) => ({ ...prev, rating: false }));
    }
  }

  return (
    <div className="mt-3.5 flex items-center gap-2 border-t border-line pt-3">
      {/* ---- icon actions ---- */}
      <div className="flex items-center gap-1">
        {ACTIONS.map(({ type, icon: Icon, label, tint }) => {
          const isActive = !!active[type];
          const isLoading = !!loading[type];
          return (
            <motion.button
              key={type}
              type="button"
              whileTap={{ scale: 0.9 }}
              onClick={() => handleAction(type)}
              title={label}
              aria-label={label}
              aria-pressed={isActive}
              className="flex h-7 w-7 items-center justify-center rounded-lg border transition-colors"
              style={{
                borderColor: isActive ? tint : 'var(--line)',
                background: isActive ? `color-mix(in srgb, ${tint} 16%, transparent)` : 'transparent',
                color: isActive ? tint : 'var(--ink-faint)',
              }}
            >
              {isLoading ? (
                <Loader2 size={13} className="animate-spin" />
              ) : (
                <Icon size={13} fill={isActive && FILLED.has(type) ? 'currentColor' : 'none'} />
              )}
            </motion.button>
          );
        })}
      </div>

      {/* ---- star rating ---- */}
      <div
        className="ml-auto flex items-center gap-0.5"
        onMouseLeave={() => setHoveredStar(0)}
      >
        {[1, 2, 3, 4, 5].map((star) => {
          const lit = star <= (hoveredStar || rating);
          return (
            <button
              key={star}
              type="button"
              onMouseEnter={() => setHoveredStar(star)}
              onClick={() => handleRating(star)}
              disabled={!!loading.rating}
              title={`Rate ${star}/5`}
              aria-label={`Rate ${star} out of 5`}
              className="p-0.5 transition-transform hover:scale-110 disabled:cursor-wait"
            >
              <Star
                size={13}
                style={{ color: lit ? 'var(--warn)' : 'var(--ink-faint)', opacity: lit ? 1 : 0.4 }}
                fill={lit ? 'currentColor' : 'none'}
              />
            </button>
          );
        })}
      </div>
    </div>
  );
}
