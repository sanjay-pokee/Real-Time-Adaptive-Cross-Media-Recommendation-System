import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Check, Lock, Mail, Sparkles, UserPlus } from 'lucide-react';
import { USERS } from '../components/UserSelector';
import GoogleSignInButton from '../components/GoogleSignInButton';
import {
  ACCOUNTS_ENABLED, apiErrorMessage, getAuthConfig, login, loginWithGoogle, signup,
} from '../api/client';
import { accountToSessionUser } from '../utils/session';

const MIN_PASSWORD_LENGTH = 8;

// Real accounts first; the demo personas stay one tab away, because the panel
// demo and the paper's scenarios run on their seeded histories.
const TABS = ACCOUNTS_ENABLED
  ? [
      { value: 'login', label: 'Log in' },
      { value: 'signup', label: 'Sign up' },
      { value: 'demo', label: 'Demo' },
    ]
  : [{ value: 'demo', label: 'Demo' }];

const PITCH = [
  { value: 'Qdrant', label: 'Vector retrieval over one unified catalogue' },
  { value: 'LightGCN', label: 'Collaborative signal from the interaction graph' },
  { value: 'EMA', label: 'Session-level preference drift, updated live' },
  { value: 'Eligibility', label: 'Age and risk checked before anything is scored' },
];

// The verticals the engine serves. Static here on purpose: the auth screen
// renders before any authenticated call, so it cannot ask /domains yet.
const VERTICALS = [
  { label: 'Entertainment', tint: 'var(--dom-entertainment)' },
  { label: 'Health', tint: 'var(--dom-health)' },
  { label: 'Industry', tint: 'var(--dom-industry)' },
  { label: 'Finance', tint: 'var(--dom-finance)' },
];

// Six profiles fit the login card, and this screen is where the four verticals
// have to be reachable - `USERS.slice(0, 6)` took the first six, which are all
// entertainment, so the health, industry and finance personas could only be
// found after logging in as someone else. Spread deliberately instead: three
// entertainment tastes, then one persona per new vertical.
const FEATURED_USER_IDS = [
  'user_scifi',
  'user_family',
  'user_music_pop',
  'user_health_caregiver',
  'user_industry_engineer',
  'user_finance_planner',
];

const FEATURED_USERS = FEATURED_USER_IDS
  .map((id) => USERS.find((user) => user.id === id))
  .filter(Boolean);

export default function AuthPage({ onAuth }) {
  const [mode, setMode] = useState(TABS[0].value);
  const [selectedUserId, setSelectedUserId] = useState(USERS[0].id);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [googleClientId, setGoogleClientId] = useState(null);

  // The Google button appears only when the backend has a client id. If the
  // backend is down this fails quietly; the email form reports it on submit.
  useEffect(() => {
    if (!ACCOUNTS_ENABLED) return undefined;
    let cancelled = false;
    getAuthConfig()
      .then((config) => {
        if (!cancelled) setGoogleClientId(config.google_client_id || null);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // The auth screen used to force `dark` regardless of preference, on the
  // reasoning that it is the product's first impression. Under the editorial
  // light surface that inverts the intent: the first impression became a dark
  // page that flipped to paper the moment anyone signed in. It now follows the
  // saved preference like every other screen, so the entrance is continuous.

  const selectedUser = useMemo(
    () => USERS.find((user) => user.id === selectedUserId) || USERS[0],
    [selectedUserId]
  );

  function switchMode(value) {
    setMode(value);
    setError('');
    setPassword('');
  }

  function handleDemo(event) {
    event.preventDefault();
    onAuth({ ...selectedUser, authType: 'demo' });
  }

  async function submitAccount(request) {
    setError('');
    setBusy(true);
    try {
      onAuth(accountToSessionUser(await request()));
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  function handleLogin(event) {
    event.preventDefault();
    submitAccount(() => login({ email, password }));
  }

  function handleSignup(event) {
    event.preventDefault();
    if (password.length < MIN_PASSWORD_LENGTH) {
      setError(`Use at least ${MIN_PASSWORD_LENGTH} characters for the password.`);
      return;
    }
    submitAccount(() => signup({ email, password, display_name: name }));
  }

  function handleGoogle(credential) {
    submitAccount(() => loginWithGoogle(credential));
  }

  return (
    <main className="above min-h-screen">
      <div className="grid min-h-screen lg:grid-cols-[1.05fr_0.95fr]">
        {/* ================= brand panel ================= */}
        <section className="relative hidden overflow-hidden border-r border-line lg:block">
          {/* Ambient light. Token-driven rather than the literal rgba values
              that used to be baked in here, which stayed indigo when the
              palette changed and no longer matched anything else. */}
          <div
            className="absolute inset-0"
            style={{
              background:
                'radial-gradient(40rem 30rem at 20% 15%, var(--field-a), transparent 65%),' +
                'radial-gradient(34rem 26rem at 85% 80%, var(--field-b), transparent 65%)',
            }}
          />

          <div className="relative z-10 flex h-full flex-col justify-between p-12 xl:p-16">
            <div className="flex items-center gap-2.5">
              <div
                className="flex h-9 w-9 items-center justify-center rounded-xl text-white"
                style={{ background: 'linear-gradient(135deg in oklab, var(--accent), var(--accent-2))' }}
              >
                <Sparkles size={17} />
              </div>
              <div className="leading-none">
                <p className="display text-base font-bold text-ink">Nexus</p>
                <p className="mt-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
                  Cross-media discovery
                </p>
              </div>
            </div>

            <div className="max-w-xl">
              <h1 className="display text-5xl font-extrabold leading-[1.04] text-ink xl:text-[3.4rem]">
                One engine.
                <br />
                <span className="text-irid">Every domain.</span>
              </h1>
              <p className="mt-6 max-w-md text-[15px] leading-relaxed text-ink-muted">
                The same retrieval stack serves entertainment, health, industry
                and finance. What changes per vertical is configuration — and an
                eligibility layer that decides what a given viewer may see
                before anything is ranked.
              </p>

              <div className="mt-7 flex flex-wrap gap-2">
                {VERTICALS.map(({ label, tint }) => (
                  <span key={label} className="chip chip-tinted" style={{ '--tint': tint }}>
                    {label}
                  </span>
                ))}
              </div>
            </div>

            <div className="grid max-w-xl gap-2.5">
              {PITCH.map(({ value, label }) => (
                <div key={value} className="panel-flat flex items-center gap-3 px-4 py-3">
                  <span className="display w-20 shrink-0 text-[13px] font-bold text-accent">
                    {value}
                  </span>
                  <span className="text-[12px] text-ink-muted">{label}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ================= form panel ================= */}
        <section className="flex min-h-screen items-center justify-center px-5 py-10 sm:px-8">
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
            className="w-full max-w-md"
          >
            <div className="mb-7 flex items-center gap-2.5 lg:hidden">
              <div
                className="flex h-9 w-9 items-center justify-center rounded-xl text-white"
                style={{ background: 'linear-gradient(135deg in oklab, var(--accent), var(--accent-2))' }}
              >
                <Sparkles size={17} />
              </div>
              <p className="display text-base font-bold text-ink">Nexus</p>
            </div>

            <div className="panel panel-lit p-6 sm:p-7">
              {/* ---- mode switch ---- */}
              {TABS.length > 1 && (
                <div
                  className="mb-6 grid gap-1 rounded-xl border border-line bg-surface-3 p-1"
                  style={{ gridTemplateColumns: `repeat(${TABS.length}, minmax(0, 1fr))` }}
                >
                  {TABS.map(({ value, label }) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => switchMode(value)}
                      aria-pressed={mode === value}
                      className="relative rounded-lg px-4 py-2 text-[13px] font-semibold transition-colors"
                      style={{ color: mode === value ? 'var(--accent-ink)' : 'var(--ink-faint)' }}
                    >
                      {/* One indicator slides between the tabs rather than
                          each tab fading its own background in and out. */}
                      {mode === value ? (
                        <motion.span
                          layoutId="auth-mode-pill"
                          transition={{ type: 'spring', stiffness: 420, damping: 34 }}
                          className="absolute inset-0 rounded-lg"
                          style={{
                            background: 'linear-gradient(120deg in oklab, var(--accent), var(--accent-2))',
                            boxShadow: 'var(--shadow-accent)',
                          }}
                        />
                      ) : null}
                      <span className="relative">{label}</span>
                    </button>
                  ))}
                </div>
              )}

              {mode === 'demo' ? (
                <form onSubmit={handleDemo} className="space-y-5">
                  <div>
                    <h2 className="display text-2xl font-bold text-ink">Try a demo profile</h2>
                    <p className="mt-1.5 text-[13px] leading-relaxed text-ink-muted">
                      Each demo profile carries its own interaction history, so the
                      ranking signals differ per user. No account needed.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <span className="label">Demo profile</span>
                    <div className="grid gap-1.5">
                      {FEATURED_USERS.map((user) => {
                        const active = selectedUserId === user.id;
                        return (
                          <button
                            key={user.id}
                            type="button"
                            onClick={() => setSelectedUserId(user.id)}
                            className="flex items-center gap-2.5 rounded-xl border px-3 py-2.5 text-left transition-colors"
                            style={{
                              borderColor: active ? 'var(--accent)' : 'var(--line)',
                              background: active ? 'var(--accent-soft)' : 'var(--surface-3)',
                            }}
                          >
                            <span
                              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-[10px] font-bold text-white"
                              style={{ background: user.accent }}
                            >
                              {user.initials}
                            </span>
                            <span className="min-w-0 flex-1">
                              <span className="block text-[13px] font-semibold text-ink">
                                {user.label}
                              </span>
                              <span className="block truncate text-[10px] text-ink-faint">
                                {user.id}
                              </span>
                            </span>
                            {active && <Check size={15} className="shrink-0 text-accent" />}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  <button type="submit" className="btn btn-primary w-full py-3">
                    Continue as {selectedUser.label}
                    <ArrowRight size={15} />
                  </button>
                </form>
              ) : (
                <form onSubmit={mode === 'login' ? handleLogin : handleSignup} className="space-y-4">
                  <div>
                    <h2 className="display text-2xl font-bold text-ink">
                      {mode === 'login' ? 'Welcome back' : 'Create your account'}
                    </h2>
                    <p className="mt-1.5 text-[13px] leading-relaxed text-ink-muted">
                      {mode === 'login'
                        ? 'Log in to pick up where you left off.'
                        : 'What you like, rate or skip is saved to your account and shapes what you are shown next.'}
                    </p>
                  </div>

                  {googleClientId && (
                    <>
                      <GoogleSignInButton
                        clientId={googleClientId}
                        text={mode === 'login' ? 'signin_with' : 'signup_with'}
                        onCredential={handleGoogle}
                        onError={setError}
                      />
                      <div className="flex items-center gap-3 text-[11px] font-medium text-ink-faint">
                        <span className="h-px flex-1 bg-line" />
                        or with email
                        <span className="h-px flex-1 bg-line" />
                      </div>
                    </>
                  )}

                  {mode === 'signup' && (
                    <Labelled label="Name" icon={UserPlus}>
                      <input
                        value={name}
                        onChange={(event) => setName(event.target.value)}
                        className="field w-full py-2.5 pl-10 pr-3"
                        placeholder="Sanjay Kumar"
                        autoComplete="name"
                        maxLength={120}
                        required
                      />
                    </Labelled>
                  )}

                  <Labelled label="Email" icon={Mail}>
                    <input
                      type="email"
                      value={email}
                      onChange={(event) => setEmail(event.target.value)}
                      className="field w-full py-2.5 pl-10 pr-3"
                      placeholder="you@example.com"
                      autoComplete="email"
                      maxLength={254}
                      required
                    />
                  </Labelled>

                  <Labelled label="Password" icon={Lock}>
                    <input
                      type="password"
                      value={password}
                      onChange={(event) => setPassword(event.target.value)}
                      className="field w-full py-2.5 pl-10 pr-3"
                      placeholder={mode === 'signup' ? `At least ${MIN_PASSWORD_LENGTH} characters` : ''}
                      autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                      maxLength={128}
                      required
                    />
                  </Labelled>

                  {error && (
                    <p
                      role="alert"
                      className="rounded-xl px-3 py-2 text-[12px] font-medium"
                      style={{
                        color: 'var(--bad)',
                        background: 'color-mix(in oklab, var(--bad) 12%, transparent)',
                      }}
                    >
                      {error}
                    </p>
                  )}

                  <button type="submit" disabled={busy} className="btn btn-primary w-full py-3">
                    {busy ? 'One moment…' : mode === 'login' ? 'Log in' : 'Create account'}
                    <ArrowRight size={15} />
                  </button>
                </form>
              )}
            </div>
          </motion.div>
        </section>
      </div>
    </main>
  );
}

function Labelled({ label, icon: Icon, children }) {
  return (
    <label className="block">
      <span className="label mb-1.5 block">{label}</span>
      <span className="relative block">
        <Icon size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-ink-faint" />
        {children}
      </span>
    </label>
  );
}
