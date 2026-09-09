import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Check, Lock, Mail, Sparkles, UserPlus } from 'lucide-react';
import { USERS } from '../components/UserSelector';

const DEMO_PASSWORD = 'demo123';

const PITCH = [
  { value: 'Qdrant', label: 'Vector retrieval over a unified catalog' },
  { value: 'LightGCN', label: 'Collaborative signal from the interaction graph' },
  { value: 'EMA', label: 'Session-level preference drift, updated live' },
];

function initialsFromName(name) {
  return (
    name
      .split(' ')
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase())
      .join('') || 'NU'
  );
}

export default function AuthPage({ onAuth }) {
  const [mode, setMode] = useState('login');
  const [selectedUserId, setSelectedUserId] = useState(USERS[0].id);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');

  // The auth screen is always dark - it is the product's first impression.
  useEffect(() => {
    const previous = document.documentElement.getAttribute('data-theme');
    document.documentElement.setAttribute('data-theme', 'dark');
    return () => {
      if (previous) document.documentElement.setAttribute('data-theme', previous);
    };
  }, []);

  const selectedUser = useMemo(
    () => USERS.find((user) => user.id === selectedUserId) || USERS[0],
    [selectedUserId]
  );

  function handleLogin(event) {
    event.preventDefault();
    setError('');
    onAuth({ ...selectedUser, authType: 'demo' });
  }

  function handleSignup(event) {
    event.preventDefault();
    setError('');
    const cleanName = name.trim();
    const cleanEmail = email.trim();

    if (!cleanName || !cleanEmail) {
      setError('Enter a name and email to create a demo account.');
      return;
    }

    onAuth({
      id: `demo_${
        cleanEmail.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || 'user'
      }`,
      label: cleanName,
      email: cleanEmail,
      initials: initialsFromName(cleanName),
      accent: '#7c8cff',
      authType: 'signup',
    });
  }

  return (
    <main className="above min-h-screen">
      <div className="grid min-h-screen lg:grid-cols-[1.05fr_0.95fr]">
        {/* ================= brand panel ================= */}
        <section className="relative hidden overflow-hidden border-r border-line lg:block">
          {/* layered ambient light instead of a flat image wash */}
          <div
            className="absolute inset-0"
            style={{
              background:
                'radial-gradient(40rem 30rem at 20% 15%, rgba(124,140,255,.22), transparent 65%),' +
                'radial-gradient(34rem 26rem at 85% 80%, rgba(167,139,250,.16), transparent 65%)',
            }}
          />

          <div className="relative z-10 flex h-full flex-col justify-between p-12 xl:p-16">
            <div className="flex items-center gap-2.5">
              <div
                className="flex h-9 w-9 items-center justify-center rounded-xl text-white"
                style={{ background: 'linear-gradient(135deg, var(--accent), var(--accent-2))' }}
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
                Find what fits
                <br />
                the mood, not just
                <br />
                the keyword.
              </h1>
              <p className="mt-6 max-w-md text-[15px] leading-relaxed text-ink-muted">
                One workspace for movies, books, and music — where every
                recommendation shows the signals that ranked it.
              </p>
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
                style={{ background: 'linear-gradient(135deg, var(--accent), var(--accent-2))' }}
              >
                <Sparkles size={17} />
              </div>
              <p className="display text-base font-bold text-ink">Nexus</p>
            </div>

            <div className="panel panel-lit p-6 sm:p-7">
              {/* ---- mode switch ---- */}
              <div className="mb-6 grid grid-cols-2 gap-1 rounded-xl border border-line bg-surface-3 p-1">
                {['login', 'signup'].map((value) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setMode(value)}
                    className="rounded-lg px-4 py-2 text-[13px] font-semibold transition-colors"
                    style={{
                      background: mode === value ? 'var(--surface)' : 'transparent',
                      color: mode === value ? 'var(--ink)' : 'var(--ink-faint)',
                      boxShadow: mode === value ? 'var(--shadow-sm)' : 'none',
                    }}
                  >
                    {value === 'login' ? 'Log in' : 'Sign up'}
                  </button>
                ))}
              </div>

              {mode === 'login' ? (
                <form onSubmit={handleLogin} className="space-y-5">
                  <div>
                    <h2 className="display text-2xl font-bold text-ink">Welcome back</h2>
                    <p className="mt-1.5 text-[13px] leading-relaxed text-ink-muted">
                      Each demo profile carries its own interaction history, so the
                      ranking signals differ per user.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <span className="label">Demo profile</span>
                    <div className="grid gap-1.5">
                      {USERS.slice(0, 6).map((user) => {
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

                  <p className="panel-flat px-3.5 py-2.5 text-[11.5px] text-ink-muted">
                    Demo password:{' '}
                    <span className="font-mono font-semibold text-ink">{DEMO_PASSWORD}</span>
                  </p>

                  <button type="submit" className="btn btn-primary w-full py-3">
                    Continue as {selectedUser.label}
                    <ArrowRight size={15} />
                  </button>
                </form>
              ) : (
                <form onSubmit={handleSignup} className="space-y-4">
                  <div>
                    <h2 className="display text-2xl font-bold text-ink">Create demo account</h2>
                    <p className="mt-1.5 text-[13px] leading-relaxed text-ink-muted">
                      Creates a local profile for this browser session.
                    </p>
                  </div>

                  <Labelled label="Name" icon={UserPlus}>
                    <input
                      value={name}
                      onChange={(event) => setName(event.target.value)}
                      className="field py-2.5 pl-10 pr-3"
                      placeholder="Sanjay Kumar"
                    />
                  </Labelled>

                  <Labelled label="Email" icon={Mail}>
                    <input
                      type="email"
                      value={email}
                      onChange={(event) => setEmail(event.target.value)}
                      className="field py-2.5 pl-10 pr-3"
                      placeholder="you@example.com"
                    />
                  </Labelled>

                  <Labelled label="Password" icon={Lock}>
                    <input
                      type="password"
                      className="field py-2.5 pl-10 pr-3"
                      placeholder="Any dummy password"
                    />
                  </Labelled>

                  {error && (
                    <p
                      className="rounded-xl px-3 py-2 text-[12px] font-medium"
                      style={{
                        color: 'var(--bad)',
                        background: 'color-mix(in srgb, var(--bad) 12%, transparent)',
                      }}
                    >
                      {error}
                    </p>
                  )}

                  <button type="submit" className="btn btn-primary w-full py-3">
                    Create account
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
