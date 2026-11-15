import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Check } from 'lucide-react';
import { apiErrorMessage, getStarterItems, postInteraction } from '../api/client';
import { coverFor, imageFor, monogram } from '../utils/cover';

const MIN_PICKS = 3;

const GROUPS = [
  ['movie', 'Movies'],
  ['show', 'Shows'],
  ['book', 'Books'],
  ['music', 'Music'],
];

/**
 * "Pick a few things you like", shown once to a new account before Home.
 *
 * A new account has no history: LightGCN has never seen it and its EMA taste
 * vector is empty, so its first results would be the same as anyone's. Each
 * pick is saved as an ordinary like, which seeds both the taste vector and
 * the interaction log the graph is later trained on.
 */
export default function OnboardingPage({ user, onDone }) {
  // null while loading, so "no picks available" is distinguishable from "not yet".
  const [items, setItems] = useState(null);
  const [picked, setPicked] = useState(() => new Set());
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getStarterItems()
      .then((data) => {
        if (!cancelled) setItems(data.items || []);
      })
      .catch((err) => {
        if (cancelled) return;
        setItems([]);
        setError(apiErrorMessage(err, 'Could not load the picks.'));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const groups = useMemo(
    () =>
      GROUPS.map(([type, label]) => [label, (items || []).filter((item) => item.content_type === type)])
        .filter(([, entries]) => entries.length > 0),
    [items],
  );

  function toggle(globalId) {
    setPicked((current) => {
      const next = new Set(current);
      if (next.has(globalId)) next.delete(globalId);
      else next.add(globalId);
      return next;
    });
  }

  async function finish() {
    setSaving(true);
    setError(null);
    try {
      // One at a time, not in parallel: each like also updates the taste
      // vector, a read-modify-write that concurrent requests could interleave.
      for (const globalId of picked) {
        await postInteraction({
          user_id: user.id,
          entity_id: globalId,
          event_type: 'like',
          context: { source: 'onboarding' },
        });
      }
      onDone(picked.size);
    } catch (err) {
      setError(apiErrorMessage(err, 'Could not save your picks. Try again.'));
      setSaving(false);
    }
  }

  const firstName = String(user.label || '').split(' ')[0];
  const remaining = Math.max(0, MIN_PICKS - picked.size);

  return (
    <div className="relative min-h-screen">
      <main className="mx-auto max-w-[1200px] px-4 pb-32 pt-10 sm:px-6">
        <motion.header
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="max-w-2xl"
        >
          <p className="label">Welcome{firstName ? `, ${firstName}` : ''}</p>
          <h1 className="display mt-2 text-[2.2rem] leading-tight text-ink sm:text-[2.8rem]">
            Pick a few things you like
          </h1>
          <p className="mt-3 text-[14px] leading-relaxed text-ink-muted">
            Choose at least {MIN_PICKS}. They are saved as likes, so your first results are
            already tuned to you instead of being the same for everyone.
          </p>
        </motion.header>

        {items === null && (
          <p className="mt-10 text-[13px] text-ink-muted" role="status">Loading picks…</p>
        )}

        {groups.map(([label, entries]) => (
          <section key={label} className="mt-9">
            <h2 className="label">{label}</h2>
            <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
              {entries.map((item) => (
                <PickCard
                  key={item.global_id}
                  item={item}
                  selected={picked.has(item.global_id)}
                  onToggle={() => toggle(item.global_id)}
                />
              ))}
            </div>
          </section>
        ))}
      </main>

      {/* Fixed so the count and Continue stay in reach while scrolling the grid. */}
      <div
        className="fixed inset-x-0 bottom-0 z-30 border-t px-4 py-3 sm:px-6"
        style={{ borderColor: 'var(--line)', background: 'color-mix(in oklab, var(--surface) 94%, transparent)' }}
      >
        <div className="mx-auto flex max-w-[1200px] flex-wrap items-center gap-3">
          <p className="min-w-0 flex-1 text-[13px] text-ink-muted" aria-live="polite">
            {error ? (
              <span role="alert" style={{ color: 'var(--bad)' }}>{error}</span>
            ) : picked.size === 0 ? (
              `Pick ${MIN_PICKS} or more to continue.`
            ) : remaining > 0 ? (
              `${picked.size} picked - ${remaining} more to go.`
            ) : (
              `${picked.size} picked.`
            )}
          </p>
          <button type="button" onClick={() => onDone(0)} disabled={saving} className="btn btn-ghost">
            Skip for now
          </button>
          <button
            type="button"
            onClick={finish}
            disabled={saving || remaining > 0}
            className="btn btn-primary disabled:cursor-not-allowed disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Continue'}
            <ArrowRight size={15} />
          </button>
        </div>
      </div>
    </div>
  );
}

function PickCard({ item, selected, onToggle }) {
  const [broken, setBroken] = useState(false);
  const src = broken ? '' : imageFor(item);
  const square = item.content_type === 'music';

  return (
    <button
      type="button"
      onClick={onToggle}
      aria-pressed={selected}
      className="group relative flex flex-col overflow-hidden rounded-xl border text-left transition-[border-color,box-shadow,transform] active:scale-[0.98]"
      style={{
        borderColor: selected ? 'var(--accent)' : 'var(--line)',
        boxShadow: selected ? '0 0 0 2px var(--accent)' : 'var(--shadow-1)',
        background: 'var(--surface)',
      }}
    >
      <span
        className={`relative block w-full ${square ? 'aspect-square' : 'aspect-[2/3]'}`}
        style={{ background: coverFor(item).background }}
      >
        {src ? (
          <img
            src={src}
            alt=""
            loading="lazy"
            onError={() => setBroken(true)}
            className="absolute inset-0 h-full w-full object-cover"
          />
        ) : (
          <span className="display absolute inset-0 flex items-center justify-center text-3xl text-white/90">
            {monogram(item.title)}
          </span>
        )}
        {selected && (
          <span
            className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-full"
            style={{ background: 'var(--accent)', color: 'var(--accent-ink)' }}
          >
            <Check size={15} strokeWidth={3} />
          </span>
        )}
      </span>
      <span className="flex flex-1 flex-col gap-1 p-2.5">
        <span className="clamp-2 text-[13px] font-semibold leading-snug text-ink">{item.title}</span>
        <span className="text-[11px] capitalize text-ink-faint">{item.genre}</span>
      </span>
    </button>
  );
}
