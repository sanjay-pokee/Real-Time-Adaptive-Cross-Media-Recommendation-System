/**
 * Static demo backend, used when the frontend is built with VITE_DEMO=1.
 *
 * The hosted demo has no API server, so it answers from demo-snapshot.json:
 * real /recommend and /recommend/item responses recorded from the live backend
 * for every demo profile and every query chip, plus a set of free-text queries.
 * Anything not recorded falls back to recorded results for the same query
 * re-filtered for the requested scope, then to a keyword match over every item
 * in the snapshot - with the same age/domain eligibility rules the backend uses.
 */

let snapshotPromise = null;

function loadSnapshot() {
  // no-cache revalidates by ETag, so a redeployed snapshot is never shadowed by a stale copy.
  snapshotPromise ??= fetch(`${import.meta.env.BASE_URL}demo-snapshot.json`, { cache: 'no-cache' }).then((res) => {
    if (!res.ok) throw new Error(`demo snapshot unavailable (${res.status})`);
    return res.json();
  });
  return snapshotPromise;
}

// A short pause so loading states render the way they do against the real API.
const settle = (value) => new Promise((resolve) => setTimeout(() => resolve(value), 180));

const norm = (text) => String(text || '').trim().toLowerCase();

function recKey(user, query, domain, age) {
  return `${user || '*'}|${norm(query)}|${domain || ''}|${age ?? ''}`;
}

function hydrate(snap, recorded) {
  return recorded.results.map((r) => ({ ...snap.items[r.global_id], ...r }));
}

function eligible(item, { domain, content_type, age, safe_mode }) {
  const effectiveAge = safe_mode ? 7 : age ?? 13;
  if (domain && item.domain !== domain) return false;
  if (content_type && item.content_type !== content_type) return false;
  const minAge = Number(item.audience_min_age);
  return !(Number.isFinite(minAge) && minAge > effectiveAge);
}

function advisoryFor(snap, domain, results) {
  const domains = snap.domains?.domains || [];
  const name = domain || (results.length && results.every((r) => r.domain === results[0].domain)
    ? results[0].domain
    : null);
  return domains.find((entry) => entry.name === name)?.advisory || null;
}

function keywordSearch(snap, query, scope) {
  const tokens = norm(query).split(/[^a-z0-9]+/).filter((t) => t.length > 1);
  if (!tokens.length) return [];
  const scored = [];
  for (const item of Object.values(snap.items)) {
    if (!eligible(item, scope)) continue;
    const title = norm(item.title);
    const cats = norm(item.categories);
    const creators = norm(item.creators);
    const desc = norm(item.description);
    let score = 0;
    for (const t of tokens) {
      if (title.includes(t)) score += 3;
      if (cats.includes(t)) score += 2;
      if (creators.includes(t)) score += 2;
      if (desc.includes(t)) score += 1;
    }
    if (score > 0) scored.push({ item, score });
  }
  const top = Math.max(1, ...scored.map((s) => s.score));
  return scored
    .sort((a, b) => b.score - a.score)
    .map(({ item, score }) => ({ ...item, score: score / top, match_kind: 'keyword', semantic_score: score / top }));
}

export async function demoRecommend({ query, user_id, top_k = 10, content_type = null, domain = null, age = null, safe_mode = false }) {
  const snap = await loadSnapshot();
  const scope = { domain, content_type, age, safe_mode };
  const recAge = safe_mode ? 7 : age;
  let results = null;

  // 1. Recorded for exactly this viewer and scope (or unpersonalised).
  const exact = snap.recommend[recKey(user_id, query, domain, recAge)]
    || snap.recommend[recKey(null, query, domain, recAge)];
  if (exact) results = hydrate(snap, exact).filter((item) => eligible(item, scope));

  // 2. Same query recorded under another scope, re-filtered for this one.
  if (!results?.length) {
    const q = norm(query);
    const candidates = Object.entries(snap.recommend)
      .filter(([key]) => key.split('|')[1] === q)
      .sort(([a], [b]) => Number(b.startsWith(`${user_id}|`)) - Number(a.startsWith(`${user_id}|`)));
    for (const [, recorded] of candidates) {
      const filtered = hydrate(snap, recorded).filter((item) => eligible(item, scope));
      if (filtered.length) { results = filtered; break; }
    }
  }

  // 3. Keyword match over every item in the snapshot.
  if (!results?.length) results = keywordSearch(snap, query, scope);

  results = results.slice(0, top_k);
  return settle({
    query, top_k, content_type, domain,
    advisory: advisoryFor(snap, domain, results),
    results,
  });
}

export async function demoSimilar({ global_id, top_k = 10, content_type = null, domain = null, age = null, safe_mode = false }) {
  const snap = await loadSnapshot();
  const scope = { domain, content_type, age, safe_mode };
  const recorded = snap.similar[`${global_id}|${safe_mode ? 7 : age ?? ''}`]
    || Object.entries(snap.similar).find(([key]) => key.startsWith(`${global_id}|`))?.[1];
  let results = recorded ? hydrate(snap, recorded).filter((item) => eligible(item, scope)) : [];
  if (!results.length) {
    const source = snap.items[global_id];
    results = source
      ? keywordSearch(snap, `${source.categories} ${source.creators}`, scope).filter((r) => r.global_id !== global_id)
      : [];
  }
  results = results.slice(0, top_k);
  return settle({ global_id, top_k, content_type, domain, advisory: advisoryFor(snap, domain, results), results });
}

export async function demoSuggest(q, limit = 8) {
  const snap = await loadSnapshot();
  const needle = norm(q);
  const seen = new Set();
  const out = [];
  const push = (label, hint, kind) => {
    const key = `${kind}|${norm(label)}`;
    if (out.length >= limit || seen.has(key)) return;
    seen.add(key);
    out.push({ label, hint, kind, query: label });
  };
  for (const item of Object.values(snap.items)) {
    if (norm(item.title).includes(needle)) push(item.title, item.content_type, 'title');
  }
  for (const item of Object.values(snap.items)) {
    for (const creator of String(item.creators || '').split(',')) {
      if (creator.trim() && norm(creator).includes(needle)) push(creator.trim(), 'creator', 'person');
    }
  }
  for (const item of Object.values(snap.items)) {
    for (const cat of String(item.categories || '').split(',')) {
      if (cat.trim() && norm(cat).includes(needle)) push(cat.trim(), 'category', 'category');
    }
  }
  return { q, suggestions: out };
}

// Interactions live in memory for the visit; there is no server to persist them.
const interactions = new Map();
const BOOLEAN_EVENTS = new Set(['view', 'like', 'bookmark', 'skip', 'complete']);

export async function demoPostInteraction({ user_id, entity_id, event_type, event_value = 1 }) {
  const key = `${user_id}|${entity_id}`;
  const state = interactions.get(key) || {};
  if (BOOLEAN_EVENTS.has(event_type)) state[event_type] = !state[event_type];
  if (event_type === 'rating') state.rating = Number(event_value) || 0;
  interactions.set(key, state);
  return settle({ status: 'recorded', user_id, entity_id, event_type });
}

export async function demoInteractionState(userId, entityId) {
  const state = interactions.get(`${userId}|${entityId}`) || {};
  return {
    user_id: userId, entity_id: entityId,
    view: false, like: false, bookmark: false, skip: false, complete: false, rating: 0,
    ...state,
  };
}

export async function demoHealth() {
  const snap = await loadSnapshot();
  return snap.health;
}

export async function demoDomains() {
  const snap = await loadSnapshot();
  return snap.domains;
}
