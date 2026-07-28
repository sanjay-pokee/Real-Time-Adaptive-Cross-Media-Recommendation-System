import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Check, Film, Library, Lock, Mail, Music2, Sparkles, UserPlus } from 'lucide-react';
import { USERS } from '../components/UserSelector';
import heroImage from '../assets/hero.png';

const DEMO_PASSWORD = 'demo123';

function initialsFromName(name) {
  return name
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map(part => part[0]?.toUpperCase())
    .join('') || 'NU';
}

export default function AuthPage({ onAuth }) {
  const [mode, setMode] = useState('login');
  const [selectedUserId, setSelectedUserId] = useState(USERS[0].id);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');

  const selectedUser = useMemo(
    () => USERS.find(user => user.id === selectedUserId) || USERS[0],
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
      id: `demo_${cleanEmail.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || 'user'}`,
      label: cleanName,
      email: cleanEmail,
      initials: initialsFromName(cleanName),
      accent: '#2563EB',
      authType: 'signup',
    });
  }

  return (
    <main className="min-h-screen bg-app text-slate-950">
      <div className="grid min-h-screen lg:grid-cols-[1.05fr_0.95fr]">
        <section className="relative hidden overflow-hidden bg-slate-950 text-white lg:block">
          <img src={heroImage} alt="Recommendation preview" className="absolute inset-0 h-full w-full object-cover opacity-35" />
          <div className="absolute inset-0 bg-[linear-gradient(135deg,rgba(2,6,23,0.98),rgba(15,23,42,0.76)_42%,rgba(37,99,235,0.5))]" />
          <div className="relative z-10 flex h-full flex-col justify-between p-12 xl:p-16">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-white text-slate-950 shadow-soft">
                <Sparkles size={20} />
              </div>
              <div>
                <p className="text-xl font-black tracking-tight">Nexus</p>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-blue-100/70">Cross-media discovery</p>
              </div>
            </div>

            <div className="max-w-xl">
              <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-1.5 text-sm font-semibold text-blue-50 backdrop-blur">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                Movies, books, and music in one recommendation workspace
              </div>
              <h1 className="font-display text-5xl font-black leading-[1.02] tracking-tight xl:text-6xl">
                Find what fits the mood, not just the keyword.
              </h1>
              <p className="mt-6 max-w-lg text-base leading-7 text-slate-200">
                Sign in as a demo user and explore personalized results ranked with semantic, graph, EMA, and final hybrid signals.
              </p>
            </div>

            <div className="grid max-w-xl grid-cols-3 gap-3">
              {[
                { icon: Film, label: 'Movies' },
                { icon: Library, label: 'Books' },
                { icon: Music2, label: 'Music' },
              ].map(({ icon: Icon, label }) => (
                <div key={label} className="rounded-2xl border border-white/12 bg-white/10 p-4 backdrop-blur">
                  <Icon size={19} className="text-blue-200" />
                  <p className="mt-3 text-sm font-bold">{label}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="flex min-h-screen items-center justify-center px-5 py-10 sm:px-8">
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="w-full max-w-md"
          >
            <div className="mb-8 flex items-center gap-3 lg:hidden">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-950 text-white">
                <Sparkles size={18} />
              </div>
              <div>
                <p className="text-lg font-black tracking-tight">Nexus</p>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Cross-media discovery</p>
              </div>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-soft sm:p-8">
              <div className="mb-6 grid grid-cols-2 rounded-2xl bg-slate-100 p-1">
                <button
                  type="button"
                  onClick={() => setMode('login')}
                  className={`rounded-xl px-4 py-2.5 text-sm font-extrabold transition ${mode === 'login' ? 'bg-white text-slate-950 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
                >
                  Log in
                </button>
                <button
                  type="button"
                  onClick={() => setMode('signup')}
                  className={`rounded-xl px-4 py-2.5 text-sm font-extrabold transition ${mode === 'signup' ? 'bg-white text-slate-950 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
                >
                  Sign up
                </button>
              </div>

              {mode === 'login' ? (
                <form onSubmit={handleLogin} className="space-y-5">
                  <div>
                    <h2 className="font-display text-3xl font-black tracking-tight text-slate-950">Welcome back</h2>
                    <p className="mt-2 text-sm leading-6 text-slate-500">Choose a dummy user profile to enter the recommendation dashboard.</p>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-extrabold text-slate-700">Demo user</label>
                    <div className="grid gap-2">
                      {USERS.slice(0, 6).map(user => (
                        <button
                          key={user.id}
                          type="button"
                          onClick={() => setSelectedUserId(user.id)}
                          className={`flex items-center gap-3 rounded-2xl border px-3 py-3 text-left transition ${selectedUserId === user.id ? 'border-blue-600 bg-blue-50' : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'}`}
                        >
                          <span className="flex h-10 w-10 items-center justify-center rounded-xl text-sm font-black text-white" style={{ background: user.accent }}>
                            {user.initials}
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="block text-sm font-black text-slate-950">{user.label}</span>
                            <span className="block truncate text-xs font-semibold text-slate-500">{user.id}</span>
                          </span>
                          {selectedUserId === user.id && <Check size={18} className="text-blue-600" />}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs font-semibold text-slate-500">
                    Demo password: <span className="font-black text-slate-900">{DEMO_PASSWORD}</span>
                  </div>

                  <button type="submit" className="btn-primary flex w-full items-center justify-center gap-2 px-5 py-3.5 text-sm">
                    Continue as {selectedUser.label}
                    <ArrowRight size={16} />
                  </button>
                </form>
              ) : (
                <form onSubmit={handleSignup} className="space-y-5">
                  <div>
                    <h2 className="font-display text-3xl font-black tracking-tight text-slate-950">Create demo account</h2>
                    <p className="mt-2 text-sm leading-6 text-slate-500">This creates a local dummy user for the current browser session.</p>
                  </div>

                  <label className="block">
                    <span className="mb-2 block text-sm font-extrabold text-slate-700">Name</span>
                    <span className="relative block">
                      <UserPlus size={17} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
                      <input value={name} onChange={event => setName(event.target.value)} className="app-input pl-11" placeholder="Sanjay Kumar" />
                    </span>
                  </label>

                  <label className="block">
                    <span className="mb-2 block text-sm font-extrabold text-slate-700">Email</span>
                    <span className="relative block">
                      <Mail size={17} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
                      <input type="email" value={email} onChange={event => setEmail(event.target.value)} className="app-input pl-11" placeholder="you@example.com" />
                    </span>
                  </label>

                  <label className="block">
                    <span className="mb-2 block text-sm font-extrabold text-slate-700">Password</span>
                    <span className="relative block">
                      <Lock size={17} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
                      <input type="password" className="app-input pl-11" placeholder="Any dummy password" />
                    </span>
                  </label>

                  {error && <p className="rounded-xl bg-red-50 px-3 py-2 text-sm font-semibold text-red-700">{error}</p>}

                  <button type="submit" className="btn-primary flex w-full items-center justify-center gap-2 px-5 py-3.5 text-sm">
                    Create account
                    <ArrowRight size={16} />
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