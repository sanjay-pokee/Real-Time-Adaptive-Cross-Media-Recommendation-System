/**
 * Deterministic cover art.
 *
 * The catalog has no poster URLs, so every card would otherwise be a wall of
 * text. We derive a stable gradient from the item id: the same item always
 * gets the same artwork, and a grid of results reads as varied rather than
 * repetitive.
 */

function hash(text) {
  let value = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    value ^= text.charCodeAt(index);
    value = Math.imul(value, 16777619);
  }
  return Math.abs(value);
}

// Hue ranges tuned per content type so a movie never looks like a book,
// while still giving each item its own identity inside that family.
const HUE_RANGE = {
  movie: [205, 265],
  book: [140, 190],
  music: [290, 345],
};

export function coverFor(item) {
  const seed = hash(String(item?.global_id || item?.title || 'nexus'));
  const [min, max] = HUE_RANGE[item?.content_type] || HUE_RANGE.movie;

  const hue = min + (seed % (max - min));
  const hue2 = hue + 28 + (seed % 18);
  const angle = 120 + (seed % 110);

  return {
    background: `linear-gradient(${angle}deg,
      hsl(${hue} 62% 52%) 0%,
      hsl(${hue2} 58% 42%) 100%)`,
    glow: `hsl(${hue} 62% 52%)`,
  };
}

/** 1-2 character monogram used on top of the cover art. */
export function monogram(title) {
  const words = String(title || '?')
    .replace(/^(the|a|an)\s+/i, '')
    .split(/\s+/)
    .filter(Boolean);

  if (words.length === 0) return '?';
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[1][0]).toUpperCase();
}
