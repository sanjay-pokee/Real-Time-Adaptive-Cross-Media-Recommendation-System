import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Home from './pages/Home';
import AuthPage from './pages/AuthPage';
import { InteractionProvider } from './context/InteractionContext';

const SESSION_KEY = 'nexus_demo_session';

export default function App() {
  const [sessionUser, setSessionUser] = useState(null);

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(SESSION_KEY);
      if (saved) setSessionUser(JSON.parse(saved));
    } catch {
      setSessionUser(null);
    }
  }, []);

  function handleAuth(user) {
    setSessionUser(user);
    window.localStorage.setItem(SESSION_KEY, JSON.stringify(user));
  }

  function handleLogout() {
    setSessionUser(null);
    window.localStorage.removeItem(SESSION_KEY);
  }

  return (
    <InteractionProvider>
      <AnimatePresence mode="wait">
        {sessionUser ? (
          <motion.div
            key="app"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.28, ease: [0.4, 0, 0.2, 1] }}
          >
            <Home authenticatedUser={sessionUser} onLogout={handleLogout} />
          </motion.div>
        ) : (
          <motion.div
            key="auth"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.28 }}
          >
            <AuthPage onAuth={handleAuth} />
          </motion.div>
        )}
      </AnimatePresence>
    </InteractionProvider>
  );
}