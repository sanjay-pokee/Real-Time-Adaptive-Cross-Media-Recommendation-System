/**
 * InteractionContext — global session-level interaction state
 *
 * Stores: { [userId::globalId]: { active: {like,bookmark,...}, rating: 0 } }
 *
 * All InteractionButtons instances read from and write to this shared context,
 * so liking/rating an item in one place (main feed, drawer, modal) is instantly
 * reflected everywhere else without any extra backend reads.
 *
 * On the first access for a new key we try to hydrate from the backend
 * GET /interactions/{userId}/{entityId} so state survives page reloads.
 */

import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { getUserInteractionState } from '../api/client';

const InteractionContext = createContext(null);

export function InteractionProvider({ children }) {
  // Map of "userId::globalId" → { active: {}, rating: 0, loaded: bool }
  const [store, setStore] = useState({});
  // Track which keys are currently being fetched to avoid double-requests
  const fetchingRef = useRef(new Set());

  /**
   * Ensure the state for a key is initialised.
   * If not yet loaded, kick off a background fetch from the backend.
   */
  const ensureLoaded = useCallback((userId, globalId) => {
    if (!userId || !globalId) return;
    const key = `${userId}::${globalId}`;
    setStore(prev => {
      if (prev[key]) return prev; // already initialised
      // Optimistically initialise with defaults so the UI never hangs
      const initialised = {
        ...prev,
        [key]: { active: {}, rating: 0, loaded: false },
      };

      // Background fetch from backend (only once per key)
      if (!fetchingRef.current.has(key)) {
        fetchingRef.current.add(key);
        getUserInteractionState(userId, globalId)
          .then(data => {
            setStore(current => ({
              ...current,
              [key]: {
                active: {
                  view: !!data.view,
                  like: !!data.like,
                  bookmark: !!data.bookmark,
                  skip: !!data.skip,
                  complete: !!data.complete,
                },
                rating: data.rating ?? 0,
                loaded: true,
              },
            }));
          })
          .catch(() => {
            // Backend may not have data (new item / no interactions) — keep defaults
            setStore(current => ({
              ...current,
              [key]: { ...current[key], loaded: true },
            }));
          })
          .finally(() => {
            fetchingRef.current.delete(key);
          });
      }

      return initialised;
    });
  }, []);

  /** Toggle an action (like, bookmark, etc.) for a key */
  const toggleActive = useCallback((userId, globalId, actionType) => {
    const key = `${userId}::${globalId}`;
    setStore(prev => {
      const entry = prev[key] || { active: {}, rating: 0, loaded: false };
      return {
        ...prev,
        [key]: {
          ...entry,
          active: {
            ...entry.active,
            [actionType]: !entry.active[actionType],
          },
        },
      };
    });
  }, []);

  /** Set a rating for a key */
  const setRating = useCallback((userId, globalId, value) => {
    const key = `${userId}::${globalId}`;
    setStore(prev => {
      const entry = prev[key] || { active: {}, rating: 0, loaded: false };
      return {
        ...prev,
        [key]: { ...entry, rating: value },
      };
    });
  }, []);

  /** Read the current state for a key */
  const getState = useCallback(
    (userId, globalId) => {
      const key = `${userId}::${globalId}`;
      return store[key] || { active: {}, rating: 0, loaded: false };
    },
    [store],
  );

  return (
    <InteractionContext.Provider value={{ ensureLoaded, toggleActive, setRating, getState }}>
      {children}
    </InteractionContext.Provider>
  );
}

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
