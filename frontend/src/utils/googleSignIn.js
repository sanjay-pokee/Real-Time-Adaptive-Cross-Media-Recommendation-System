/**
 * Sign in with Google (Google Identity Services), loaded on demand.
 *
 * The button hands the page an ID token; the backend verifies it with Google
 * (POST /auth/google), so nothing here is trusted on its own.
 */

const SCRIPT_SRC = 'https://accounts.google.com/gsi/client';

let scriptPromise = null;
let initializedFor = null;
let credentialHandler = null;

function loadScript() {
  scriptPromise ??= new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = SCRIPT_SRC;
    script.async = true;
    script.onload = () => resolve(window.google);
    script.onerror = () => {
      // Allow a later mount to try again, e.g. once the network is back.
      scriptPromise = null;
      script.remove();
      reject(new Error('Google sign-in could not load.'));
    };
    document.head.appendChild(script);
  });
  return scriptPromise;
}

export async function renderGoogleButton(element, { clientId, text, onCredential }) {
  const google = await loadScript();
  credentialHandler = onCredential;
  // initialize() once per client id: a second call logs a warning and resets
  // Google's state. The callback goes through `credentialHandler`, so the
  // button mounted now still receives the credential.
  if (initializedFor !== clientId) {
    google.accounts.id.initialize({
      client_id: clientId,
      callback: (response) => credentialHandler?.(response.credential),
      ux_mode: 'popup',
    });
    initializedFor = clientId;
  }
  if (!element?.isConnected) return;

  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  element.replaceChildren();
  google.accounts.id.renderButton(element, {
    type: 'standard',
    theme: dark ? 'filled_black' : 'outline',
    size: 'large',
    shape: 'pill',
    text,
    // Google draws the button at a fixed pixel width between 200 and 400.
    width: Math.min(400, Math.max(200, Math.round(element.getBoundingClientRect().width))),
  });
}
