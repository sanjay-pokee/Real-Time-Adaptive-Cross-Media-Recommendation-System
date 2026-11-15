import { useCallback, useContext, useEffect } from 'react';
import { InteractionContext } from './interactionContextObject';

/**
 * Hook for InteractionButtons.
 * Returns { active, rating, isLoaded } + setters that update global context.
 */
export function useInteractionState(userId, globalId) {
  const ctx = useContext(InteractionContext);
  if (!ctx) throw new Error('useInteractionState must be used inside <InteractionProvider>');

  const { ensureLoaded, toggleActive, setRating, getState } = ctx;

  // Hydrate after commit, never during render — calling ensureLoaded inline
  // sets state on the provider while this component is still rendering, which
  // React reports as a setState-in-render warning. It is idempotent, so
  // running it in an effect keyed on the pair is equivalent and legal.
  useEffect(() => {
    ensureLoaded(userId, globalId);
  }, [ensureLoaded, userId, globalId]);

  const state = getState(userId, globalId);

  const handleToggle = useCallback(
    actionType => {
      toggleActive(userId, globalId, actionType);
    },
    [toggleActive, userId, globalId],
  );

  const handleSetRating = useCallback(
    value => {
      setRating(userId, globalId, value);
    },
    [setRating, userId, globalId],
  );

  return {
    active: state.active,
    rating: state.rating,
    isLoaded: state.loaded,
    toggleActive: handleToggle,
    setRating: handleSetRating,
  };
}
