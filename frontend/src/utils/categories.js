/**
 * "Fiction, Classics, fiction" -> ["Fiction", "Classics"].
 *
 * Catalogue categories can repeat a tag (Open Library lists some books under
 * "Fiction" twice), which showed the chip twice and gave React two children
 * with one key. Repeats are dropped ignoring case; the first spelling is kept.
 */
export function splitCategories(value, limit = Infinity) {
  const seen = new Set();
  const out = [];
  for (const entry of String(value || '').split(/[,|;]+/)) {
    const tag = entry.trim();
    const key = tag.toLowerCase();
    if (!tag || seen.has(key)) continue;
    seen.add(key);
    out.push(tag);
    if (out.length >= limit) break;
  }
  return out;
}
