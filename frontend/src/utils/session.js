// Kept in sync with --irid-1 by hand: this value is stored on the session
// object, so it cannot be a CSS variable.
const ACCOUNT_ACCENT = '#4dd9ff';

export function initialsFromName(name) {
  return (
    String(name || '')
      .split(' ')
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase())
      .join('') || 'NU'
  );
}

/**
 * The session object the app runs on, built from an /auth response.
 *
 * `authType: 'account'` marks a real, server-checked login; demo personas
 * carry `'demo'` and never touch /auth.
 */
export function accountToSessionUser(account) {
  return {
    id: account.user_id,
    label: account.display_name,
    email: account.email,
    initials: initialsFromName(account.display_name),
    accent: ACCOUNT_ACCENT,
    authType: 'account',
  };
}
