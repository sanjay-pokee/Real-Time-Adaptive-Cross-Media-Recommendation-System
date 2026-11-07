import axios from 'axios';
import {
  demoDomains, demoHealth, demoInteractionState, demoPostInteraction,
  demoRecommend, demoSimilar, demoSuggest,
} from './demo';

// `VITE_DEMO=1` builds the hosted static demo, which answers from recorded
// backend responses instead of a live API (see ./demo.js).
const DEMO = import.meta.env.VITE_DEMO === '1';

// Accounts need the live API; the static demo build offers the personas only.
export const ACCOUNTS_ENABLED = !DEMO;

// Use Vite proxy (/api → http://127.0.0.1:8000) to avoid CORS entirely.
// Falls back to direct URL if env var is set explicitly.
const BASE_URL = import.meta.env.VITE_API_BASE_URL
  ? import.meta.env.VITE_API_BASE_URL
  : '/api';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
  // Carries the session cookie when VITE_API_BASE_URL is another origin;
  // same-origin /api requests send it anyway.
  withCredentials: true,
});

/** One readable sentence for a failed call, for showing next to a form. */
export function apiErrorMessage(error, fallback = 'Something went wrong. Try again.') {
  const detail = error?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  // FastAPI validation errors arrive as a list of { msg, loc }.
  if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg;
  if (error?.request && !error?.response) return 'Could not reach the server. Is the backend running?';
  return fallback;
}

// ---- accounts (backend/auth.py) ----

/** Returns { google_client_id } - null when Google sign-in is not set up. */
export async function getAuthConfig() {
  if (DEMO) return { google_client_id: null };
  const res = await api.get('/auth/config');
  return res.data;
}

export async function signup({ email, password, display_name }) {
  const res = await api.post('/auth/signup', { email, password, display_name });
  return res.data;
}

export async function login({ email, password }) {
  const res = await api.post('/auth/login', { email, password });
  return res.data;
}

export async function loginWithGoogle(credential) {
  const res = await api.post('/auth/google', { credential });
  return res.data;
}

export async function logout() {
  const res = await api.post('/auth/logout');
  return res.data;
}

/** The logged-in account; rejects with a 401 when the session is gone. */
export async function getMe() {
  const res = await api.get('/auth/me');
  return res.data;
}

export async function checkHealth() {
  if (DEMO) return demoHealth();
  const res = await api.get('/health');
  return res.data;
}

/**
 * The domain packs this deployment serves.
 *
 * Fetched rather than hardcoded: the backend builds them from
 * config/domains.yaml, so a vertical added there shows up in the UI without a
 * frontend change. That is the whole architectural claim, so the UI should not
 * quietly contradict it by keeping its own list.
 *
 * Returns: { domains: [{ name, label, description, content_types, risk_tier,
 *            advisory }], content_types: [...] }
 */
export async function getDomains() {
  if (DEMO) return demoDomains();
  const res = await api.get('/domains');
  return res.data;
}

export async function getRecommendations({
  query,
  user_id,
  top_k = 10,
  content_type = null,
  domain = null,
  age = null,
  safe_mode = false,
}) {
  if (DEMO) return demoRecommend({ query, user_id, top_k, content_type, domain, age, safe_mode });
  const res = await api.post('/recommend', {
    query, user_id, top_k, content_type, domain, age, safe_mode,
  });
  return res.data;
}

export async function getSimilarItems({
  global_id,
  user_id,
  top_k = 10,
  content_type = null,
  domain = null,
  age = null,
  safe_mode = false,
}) {
  if (DEMO) return demoSimilar({ global_id, user_id, top_k, content_type, domain, age, safe_mode });
  const res = await api.post('/recommend/item', {
    global_id, user_id, top_k, content_type, domain, age, safe_mode,
  });
  return res.data;
}

export async function postInteraction({ user_id, entity_id, event_type, event_value = 1, context = {} }) {
  if (DEMO) return demoPostInteraction({ user_id, entity_id, event_type, event_value });
  const res = await api.post('/interactions', { user_id, entity_id, event_type, event_value, context });
  return res.data;
}

/**
 * Fetch the persisted interaction state for a user+entity pair.
 * Returns: { view, like, bookmark, skip, complete, rating }
 */
export async function getUserInteractionState(userId, entityId) {
  if (DEMO) return demoInteractionState(userId, entityId);
  // entity_id can contain slashes (e.g. "movie/123") so we encode it
  const res = await api.get(`/interactions/${encodeURIComponent(userId)}/${encodeURIComponent(entityId)}`);
  return res.data;
}

/**
 * Fetch autocomplete suggestions for a partial query.
 * Returns: { q, suggestions: [{ label, hint, kind, query }] }
 */
export async function getSuggestions(q, limit = 8) {
  if (DEMO) return demoSuggest(q, limit);
  const res = await api.get('/search/suggest', { params: { q, limit } });
  return res.data;
}

export default api;
