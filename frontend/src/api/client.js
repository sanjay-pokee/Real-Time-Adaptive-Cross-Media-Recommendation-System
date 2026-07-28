import axios from 'axios';

// Use Vite proxy (/api → http://127.0.0.1:8000) to avoid CORS entirely.
// Falls back to direct URL if env var is set explicitly.
const BASE_URL = import.meta.env.VITE_API_BASE_URL
  ? import.meta.env.VITE_API_BASE_URL
  : '/api';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

export async function checkHealth() {
  const res = await api.get('/health');
  return res.data;
}

export async function getRecommendations({ query, user_id, top_k = 10, content_type = null }) {
  const res = await api.post('/recommend', { query, user_id, top_k, content_type });
  return res.data;
}

export async function getSimilarItems({ global_id, user_id, top_k = 10, content_type = null }) {
  const res = await api.post('/recommend/item', { global_id, user_id, top_k, content_type });
  return res.data;
}

export async function postInteraction({ user_id, entity_id, event_type, event_value = 1, context = {} }) {
  const res = await api.post('/interactions', { user_id, entity_id, event_type, event_value, context });
  return res.data;
}

/**
 * Fetch the persisted interaction state for a user+entity pair.
 * Returns: { view, like, bookmark, skip, complete, rating }
 */
export async function getUserInteractionState(userId, entityId) {
  // entity_id can contain slashes (e.g. "movie/123") so we encode it
  const res = await api.get(`/interactions/${encodeURIComponent(userId)}/${encodeURIComponent(entityId)}`);
  return res.data;
}

export default api;
