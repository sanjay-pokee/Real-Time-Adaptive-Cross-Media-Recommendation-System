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

// Hue ranges tuned per content type so a movie never looks like a book, while
// still giving each item its own identity inside that family. Every content
// type in config/domains.yaml needs an entry: without one, health, industrial
// and finance rows all fell through to the movie palette and the four domains
// were indistinguishable at a glance.
const HUE_RANGE = {
  movie:      [200, 250],
  book:       [135, 175],
  music:      [315, 350],
  health:     [165, 195],
  industrial: [30, 60],
  finance:    [75, 110],
};

const DEFAULT_HUE_RANGE = [200, 250];

export function coverFor(item) {
  const seed = hash(String(item?.global_id || item?.title || 'nexus'));
  const [min, max] = HUE_RANGE[item?.content_type] || DEFAULT_HUE_RANGE;

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

/**
 * The real cover image for an item, when the catalogue has one.
 *
 * Books carry a Google Books thumbnail and the Amazon-sourced verticals carry
 * a product photo, so most of the catalogue has real art. Movies and music
 * have no image source in their datasets and fall back to the generated
 * gradient, which is why coverFor still exists.
 *
 * Google Books serves its thumbnails over plain http, which a https page
 * blocks as mixed content; rewriting the scheme is enough to make them load.
 */
export function imageFor(item) {
  const url = String(item?.image_url || '').trim();
  if (!url) return '';
  return url.startsWith('http://') ? `https://${url.slice('http://'.length)}` : url;
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
