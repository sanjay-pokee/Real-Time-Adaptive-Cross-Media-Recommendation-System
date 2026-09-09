import { useState } from 'react';
import Home from './pages/Home';
import AuthPage from './pages/AuthPage';
import { InteractionProvider } from './context/InteractionContext';

const SESSION_KEY = 'nexus_demo_session';

/**
 * Read the saved session synchronously.
 *
 * This must not happen in an effect: doing so renders the auth screen first and
 * then immediately swaps it out, which reintroduces the mount/unmount race this
 * component exists to avoid.
 */
function readSession() {
  try {
    const saved = window.localStorage.getItem(SESSION_KEY);
    return saved ? JSON.parse(saved) : null;
  } catch {
    return null;
  }
}

/**
 * The auth <-> app swap is a plain conditional render, deliberately.
 *
 * Wrapping it in AnimatePresence made the page transition depend on framer
 * motion reporting an exit animation complete, and under React StrictMode that
 * callback does not always fire - which left either the login screen or the
 * logged-in app stuck on screen permanently. Each page animates its own content
 * in on mount, so the entrance still feels alive with no exit to coordinate.
 */
export default function App() {
  const [sessionUser, setSessionUser] = useState(readSession);

  function handleAuth(user) {
    setSessionUser(user);
    try {
      window.localStorage.setItem(SESSION_KEY, JSON.stringify(user));
    } catch {
      /* storage unavailable - the session still lives in memory */
    }
  }

  function handleLogout() {
    setSessionUser(null);
    try {
      window.localStorage.removeItem(SESSION_KEY);
    } catch {
      /* nothing to clean up */
    }
  }

  return (
    <InteractionProvider>
      {sessionUser ? (
        <Home authenticatedUser={sessionUser} onLogout={handleLogout} />
      ) : (
        <AuthPage onAuth={handleAuth} />
      )}
    </InteractionProvider>
  );
}
