/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Plus Jakarta Sans', 'Inter', 'sans-serif'],
      },
      // Every colour resolves to a CSS variable, so the light/dark
      // themes in globals.css switch the whole UI with one attribute.
      colors: {
        bg: 'var(--bg)',
        surface: {
          DEFAULT: 'var(--surface)',
          2: 'var(--surface-2)',
          3: 'var(--surface-3)',
          hover: 'var(--surface-hover)',
        },
        line: {
          DEFAULT: 'var(--line)',
          strong: 'var(--line-strong)',
        },
        ink: {
          DEFAULT: 'var(--ink)',
          muted: 'var(--ink-muted)',
          faint: 'var(--ink-faint)',
        },
        accent: {
          DEFAULT: 'var(--accent)',
          2: 'var(--accent-2)',
          soft: 'var(--accent-soft)',
          ink: 'var(--accent-ink)',
        },
        sig: {
          semantic: 'var(--sig-semantic)',
          graph: 'var(--sig-graph)',
          ema: 'var(--sig-ema)',
          kg: 'var(--sig-kg)',
          profile: 'var(--sig-profile)',
        },
        type: {
          movie: 'var(--type-movie)',
          book: 'var(--type-book)',
          music: 'var(--type-music)',
        },
        ok: 'var(--ok)',
        warn: 'var(--warn)',
        bad: 'var(--bad)',
      },
      borderRadius: {
        DEFAULT: 'var(--radius)',
        lg: 'var(--radius-lg)',
      },
      boxShadow: {
        sm: 'var(--shadow-sm)',
        DEFAULT: 'var(--shadow)',
        lg: 'var(--shadow-lg)',
        accent: 'var(--shadow-accent)',
      },
      animation: {
        'float': 'float 7s ease-in-out infinite',
        'shimmer': 'shimmer 1.6s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
    },
  },
  plugins: [],
}
