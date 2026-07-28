let id = 0;

export function createToast(message, type = 'info', title = '', duration = 4000) {
  return { id: ++id, message, type, title, duration };
}