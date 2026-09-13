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
      {/* The iridescent field the glass refracts. Behind both pages so it
          survives the auth swap without restarting its animation, and purely
          decorative, so it is hidden from assistive tech. */}
      <div className="ambient-field" aria-hidden="true">
        <span className="field-blob field-a" />
        <span className="field-blob field-b" />
        <span className="field-blob field-c" />
        <span className="field-blob field-d" />
      </div>

      {/*
        Refraction. Frost alone (backdrop-filter: blur) reads as plastic; what
        makes glass look like glass is that it *bends* what is behind it. CSS
        has no lensing primitive, but backdrop-filter accepts an SVG filter
        reference, so a displacement map perturbs the backdrop per-pixel.

        The map is a low-frequency fractal turbulence, which gives a slow
        organic warp rather than a regular ripple. Scale is deliberately small
        (12): past roughly 20 the text behind a panel starts to smear into
        something that reads as a rendering fault.

        Applied only to `.lg-refract` — the hero — never to every panel. Each
        displaced backdrop is a full-size offscreen pass, and doing that to
        dozens of cards drops the frame rate on integrated graphics.
      */}
      <svg aria-hidden="true" focusable="false"
           style={{ position: 'absolute', width: 0, height: 0, pointerEvents: 'none' }}>
        <filter id="lg-refraction" x="0%" y="0%" width="100%" height="100%"
                colorInterpolationFilters="sRGB">
          <feTurbulence type="fractalNoise" baseFrequency="0.008 0.012"
                        numOctaves="2" seed="7" result="noise" />
          <feGaussianBlur in="noise" stdDeviation="2" result="softNoise" />
          <feDisplacementMap in="SourceGraphic" in2="softNoise" scale="12"
                             xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </svg>

      {sessionUser ? (
        <Home authenticatedUser={sessionUser} onLogout={handleLogout} />
      ) : (
        <AuthPage onAuth={handleAuth} />
      )}
    </InteractionProvider>
  );
}
