import { Baby, GraduationCap, ShieldAlert, UserRound } from 'lucide-react';

/**
 * The audience rating a catalogue row carries.
 *
 * These come from preprocessing/audience_tagging.py, which derives them from
 * category keywords rather than a certification body. The tooltip says so —
 * a badge that looks like an official rating when it is a heuristic is worse
 * than no badge at all.
 */
const LEVELS = {
  all_ages:   { label: 'All ages', short: 'A',   icon: Baby,          tint: 'var(--mat-all_ages)' },
  teen:       { label: '13+',      short: '13',  icon: GraduationCap, tint: 'var(--mat-teen)' },
  adult:      { label: '18+',      short: '18',  icon: UserRound,     tint: 'var(--mat-adult)' },
  restricted: { label: '21+',      short: '21',  icon: ShieldAlert,   tint: 'var(--mat-restricted)' },
};

export default function MaturityBadge({ maturity, minAge, compact = false }) {
  const level = LEVELS[maturity];
  if (!level) return null;

  const { label, short, icon: Icon, tint } = level;
  const title =
    `Audience rating: ${label}` +
    (Number.isFinite(minAge) ? ` (min age ${minAge})` : '') +
    ' — derived from category keywords, not a certified rating.';

  return (
    <span
      className="chip chip-tinted"
      style={{ '--tint': tint }}
      title={title}
    >
      <Icon size={11} aria-hidden="true" />
      {compact ? short : label}
    </span>
  );
}
