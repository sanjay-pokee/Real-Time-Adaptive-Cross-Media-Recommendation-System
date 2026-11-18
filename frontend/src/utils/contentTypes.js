import {
  BookOpen,
  Factory,
  Film,
  HeartPulse,
  LayoutGrid,
  Music,
  Tv,
  Landmark,
} from 'lucide-react';

/**
 * How each content type is shown: plural label, icon and tint. Types come from
 * /domains, so a new one needs only an entry here — and falls back to a
 * neutral pair if even that is missing.
 */
export const TYPE_STYLE = {
  movie:      { label: 'Movies',     icon: Film,       tint: 'var(--type-movie)' },
  show:       { label: 'Shows',      icon: Tv,         tint: 'var(--type-show)' },
  book:       { label: 'Books',      icon: BookOpen,   tint: 'var(--type-book)' },
  music:      { label: 'Music',      icon: Music,      tint: 'var(--type-music)' },
  health:     { label: 'Health',     icon: HeartPulse, tint: 'var(--type-health)' },
  industrial: { label: 'Industrial', icon: Factory,    tint: 'var(--type-industrial)' },
  finance:    { label: 'Finance',    icon: Landmark,   tint: 'var(--type-finance)' },
};

export const FALLBACK_TYPE_STYLE = { icon: LayoutGrid, tint: 'var(--accent)' };

/** The plural label a type is shown under ("Movies", "Music"). */
export function contentTypeLabel(name) {
  return TYPE_STYLE[name]?.label || String(name).charAt(0).toUpperCase() + String(name).slice(1);
}
