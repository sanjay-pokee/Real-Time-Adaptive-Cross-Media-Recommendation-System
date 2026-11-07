import { useEffect, useRef } from 'react';
import { renderGoogleButton } from '../utils/googleSignIn';

/**
 * Google's own button, drawn into this box by Google's script. `text` is one
 * of Google's labels: 'signin_with' or 'signup_with'.
 */
export default function GoogleSignInButton({ clientId, text, onCredential, onError }) {
  const boxRef = useRef(null);
  // The latest handlers, read when Google calls back - not the ones captured
  // when the button was drawn.
  const handlers = useRef({ onCredential, onError });
  useEffect(() => {
    handlers.current = { onCredential, onError };
  });

  useEffect(() => {
    let cancelled = false;
    renderGoogleButton(boxRef.current, {
      clientId,
      text,
      onCredential: (credential) => handlers.current.onCredential(credential),
    }).catch((error) => {
      if (!cancelled) handlers.current.onError?.(error.message);
    });
    return () => {
      cancelled = true;
    };
  }, [clientId, text]);

  return <div ref={boxRef} className="flex min-h-[44px] w-full justify-center" />;
}
