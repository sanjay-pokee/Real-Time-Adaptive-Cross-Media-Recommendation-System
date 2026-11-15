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
    hasGoogle: Boolean(account.has_google),
    interactionCount: account.interaction_count ?? 0,
  };
}

const ONBOARDED_KEY = 'nexus_onboarded';

function onboardedIds() {
  try {
    return JSON.parse(window.localStorage.getItem(ONBOARDED_KEY) || '[]');
  } catch {
    return [];
  }
}

/**
 * Whether to offer "pick a few things you like" before the first search.
 *
 * Only to a real account with nothing recorded yet, and once per browser:
 * someone who skips it is not asked again on every reload.
 */
export function needsOnboarding(user) {
  if (user?.authType !== 'account' || user.interactionCount > 0) return false;
  return !onboardedIds().includes(user.id);
}

export function markOnboarded(userId) {
  try {
    const ids = onboardedIds().filter((id) => id !== userId);
    window.localStorage.setItem(ONBOARDED_KEY, JSON.stringify([...ids, userId]));
  } catch {
    /* storage unavailable - the step may show again next visit, which is harmless */
  }
}
