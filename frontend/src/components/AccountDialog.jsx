import { useEffect, useRef, useState } from 'react';
import { Trash2, X } from 'lucide-react';
import { apiErrorMessage, deleteAccount } from '../api/client';

/**
 * The header's account chip, and the dialog it opens: who is signed in, what
 * is stored about them, and the way to remove all of it.
 *
 * A native <dialog> opened with showModal(), deliberately, not a custom
 * overlay: the browser makes the page inert, keeps focus inside, closes on Esc
 * or the back gesture, and returns focus to the chip that opened it.
 *
 * The chip opens the dialog directly rather than through an `open` prop.
 * Esc closes it outside React, and mirroring that back into state through the
 * close event left a window where the chip set `open` to a value React
 * already held, so a click right after Esc did nothing.
 */
export default function AccountDialog({ user, chip, onDeleted }) {
  const ref = useRef(null);
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  // However it closes - Esc, back gesture, click outside, the X - the next
  // open starts from the first step.
  useEffect(() => {
    const dialog = ref.current;
    const reset = () => {
      setConfirming(false);
      setError(null);
    };
    dialog?.addEventListener('close', reset);
    return () => dialog?.removeEventListener('close', reset);
  }, []);

  async function handleDelete() {
    setBusy(true);
    setError(null);
    try {
      await deleteAccount();
      ref.current?.close();
      onDeleted();
    } catch (err) {
      setError(apiErrorMessage(err, 'Could not delete the account. Try again.'));
      setBusy(false);
    }
  }

  return (
    <>
      {/* Visible on phones too (initials only): it is the way to the account. */}
      <button
        type="button"
        onClick={() => ref.current?.showModal()}
        className="flex items-center gap-2 rounded-xl border border-line bg-surface-3 px-2 py-1.5 transition-colors hover:border-line-strong"
        title="Your account"
      >
        <span
          className="flex h-6 w-6 items-center justify-center rounded-lg text-[9px] font-bold text-white"
          style={{ background: chip.accent }}
        >
          {chip.initials}
        </span>
        <span className="hidden text-[12px] font-semibold text-ink md:inline">{chip.label}</span>
      </button>

      <dialog ref={ref} className="dialog-sheet" aria-labelledby="account-dialog-title" closedby="any">
        <div className="flex items-center justify-between gap-3 border-b px-5 py-4" style={{ borderColor: 'var(--line)' }}>
          <h2 id="account-dialog-title" className="display text-xl text-ink">Your account</h2>
          <button
            type="button"
            onClick={() => ref.current?.close()}
            className="btn btn-ghost h-8 w-8 p-0"
            aria-label="Close"
          >
            <X size={15} />
          </button>
        </div>

        <div className="space-y-5 px-5 py-5">
          <div className="flex items-center gap-3">
            <span
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-[13px] font-bold text-white"
              style={{ background: user.accent }}
            >
              {user.initials}
            </span>
            <div className="min-w-0">
              <p className="truncate text-[14px] font-semibold text-ink">{user.label}</p>
              <p className="truncate text-[12.5px] text-ink-muted">{user.email}</p>
              <p className="text-[11.5px] text-ink-faint">
                {user.hasGoogle ? 'Signs in with Google' : 'Signs in with email and password'}
              </p>
            </div>
          </div>

          <section>
            <h3 className="label">What is saved</h3>
            <ul className="mt-2 list-disc space-y-1 pl-5 text-[12.5px] leading-relaxed text-ink-muted">
              <li>Your name and email, to sign you in.</li>
              <li>What you like, rate, skip or open, which tunes your results.</li>
              <li>Nothing is shared or sold, and a Google sign-in gives only your name and email.</li>
            </ul>
          </section>

          <section className="border-t pt-4" style={{ borderColor: 'var(--line)' }}>
            {confirming ? (
              <div>
                <p className="text-[13px] font-semibold text-ink">Delete your account?</p>
                <p className="mt-1 text-[12.5px] leading-relaxed text-ink-muted">
                  This permanently removes your account, every like, rating and view, and your taste
                  profile. It cannot be undone.
                </p>
                {error && (
                  <p role="alert" className="mt-2 text-[12px] font-medium" style={{ color: 'var(--bad)' }}>
                    {error}
                  </p>
                )}
                {/* Right-aligned, while the button that led here sits on the
                    left: a double click cannot land on "Delete permanently". */}
                <div className="mt-4 flex flex-wrap justify-end gap-2">
                  {/* React focuses this on mount: the safe choice is the default. */}
                  <button
                    type="button"
                    autoFocus
                    onClick={() => setConfirming(false)}
                    disabled={busy}
                    className="btn"
                  >
                    Cancel
                  </button>
                  <button type="button" onClick={handleDelete} disabled={busy} className="btn btn-danger">
                    <Trash2 size={14} />
                    {busy ? 'Deleting…' : 'Delete permanently'}
                  </button>
                </div>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setConfirming(true)}
                className="btn btn-ghost -ml-3"
                style={{ color: 'var(--bad)' }}
              >
                <Trash2 size={14} />
                Delete account…
              </button>
            )}
          </section>
        </div>
      </dialog>
    </>
  );
}
