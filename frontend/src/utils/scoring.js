/**
 * Ranking-score decomposition.
 *
 * The backend fuses signals additively (see backend/qdrant_recommender.py):
 *
 *   final = semantic
 *         + LIGHTGCN_WEIGHT * graph      (backend/settings.py, default 0.20)
 *         + EMA_WEIGHT      * ema        (backend/settings.py, default 0.15)
 *         + KG_WEIGHT       * kg         (backend/knowledge_graph.py, default 0.08)
 *         + personalisation bonus        (category / content-type affinity)
 *
 * Because it is a plain weighted sum we can show exactly how many points each
 * signal contributed to the number the ranking actually used - which is a far
 * more honest explanation than plotting the raw signal values side by side.
 */

export const WEIGHTS = {
  semantic: 1.0,
  graph: 0.20,
  ema: 0.15,
  kg: 0.08,
};

export const SIGNALS = [
  {
    key: 'semantic',
    field: 'semantic_score',
    label: 'Semantic',
    color: 'var(--sig-semantic)',
    weight: WEIGHTS.semantic,
    what: 'Sentence-BERT similarity between your query and the item text.',
    whyOff: 'Exact title/creator match - retrieved without vector search.',
  },
  {
    key: 'graph',
    field: 'graph_score',
    label: 'Graph',
    color: 'var(--sig-graph)',
    weight: WEIGHTS.graph,
    what: 'LightGCN collaborative signal from the user-item interaction graph.',
    whyOff: 'No trained LightGCN artifact loaded for this catalog yet.',
  },
  {
    key: 'ema',
    field: 'ema_score',
    label: 'EMA',
    color: 'var(--sig-ema)',
    weight: WEIGHTS.ema,
    what: 'Exponential moving average of your in-session preference drift.',
    whyOff: 'Needs interactions this session - like or rate a few items.',
  },
  {
    key: 'kg',
    field: 'kg_score',
    label: 'Knowledge graph',
    color: 'var(--sig-kg)',
    weight: WEIGHTS.kg,
    what: 'Proximity between query entities and item entities in the catalog graph.',
    whyOff: 'No shared entities between the query and this item.',
  },
];

const num = (value) => {
  if (value === null || value === undefined || value === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

/**
 * Decompose one result into per-signal point contributions.
 *
 * Returns { final, parts, active, inactive, residual } where each part is
 * { key, label, color, raw, weight, points, share } and `share` is the
 * fraction of the final score that part is responsible for (0..1).
 */
export function decomposeScore(result) {
  const final = num(result?.score) ?? 0;

  const parts = [];
  const inactive = [];

  for (const signal of SIGNALS) {
    const raw = num(result?.[signal.field]);
    if (raw === null || raw === 0) {
      inactive.push({ ...signal, raw });
      continue;
    }
    parts.push({
      key: signal.key,
      label: signal.label,
      color: signal.color,
      what: signal.what,
      raw,
      weight: signal.weight,
      points: raw * signal.weight,
    });
  }

  // Anything the named signals do not account for is the personalisation
  // re-rank bonus (category / content-type affinity, repeat-item penalties).
  const accounted = parts.reduce((sum, part) => sum + part.points, 0);
  const residual = final - accounted;

  if (Math.abs(residual) >= 0.005) {
    parts.push({
      key: 'profile',
      label: 'Profile',
      color: 'var(--sig-profile)',
      what: 'Personalisation re-rank from your category and content-type affinity.',
      raw: null,
      weight: null,
      points: residual,
    });
  }

  // Share is computed over positive magnitude so the stacked bar always fills.
  const positiveTotal = parts.reduce(
    (sum, part) => sum + Math.max(part.points, 0),
    0
  ) || 1;

  for (const part of parts) {
    part.share = Math.max(part.points, 0) / positiveTotal;
  }

  parts.sort((a, b) => b.points - a.points);

  return { final, parts, inactive, residual };
}

// Acronyms must keep their casing when a label is dropped into prose.
const ACRONYMS = new Set(['EMA', 'KG']);

const inProse = (label) =>
  ACRONYMS.has(label) ? label : label.toLowerCase();

/** Short human sentence describing why this item ranked where it did. */
export function explainRank(result) {
  const { parts } = decomposeScore(result);
  if (parts.length === 0) return 'Ranked by direct catalog match.';

  const lead = parts[0];
  const support = parts.slice(1).filter((part) => part.points > 0.001);

  const pct = Math.round(lead.share * 100);
  let sentence = `${pct}% of this score is ${inProse(lead.label)} match`;

  if (support.length === 1) {
    sentence += `, lifted by ${inProse(support[0].label)}`;
  } else if (support.length > 1) {
    const names = support.map((part) => inProse(part.label));
    sentence += `, lifted by ${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
  }

  return `${sentence}.`;
}
