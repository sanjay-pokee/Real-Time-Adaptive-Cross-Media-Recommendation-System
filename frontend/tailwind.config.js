/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ['Plus Jakarta Sans', 'Inter', 'sans-serif'],
      },
      colors: {
        glass: {
          50:  'rgba(255,255,255,0.05)',
          100: 'rgba(255,255,255,0.08)',
          200: 'rgba(255,255,255,0.12)',
          300: 'rgba(255,255,255,0.18)',
          border: 'rgba(255,255,255,0.10)',
        },
        neon: {
          blue:   '#4FC3F7',
          purple: '#CE93D8',
          pink:   '#F48FB1',
          cyan:   '#80DEEA',
          green:  '#A5D6A7',
          amber:  '#FFE082',
          orange: '#FFAB91',
        },
        dark: {
          900: '#030712',
          800: '#060d1f',
          700: '#0a1628',
          600: '#0f1f38',
        },
      },
      backdropBlur: {
        xs: '2px',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'spin-slow': 'spin 8s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-12px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        glow: {
          '0%': { boxShadow: '0 0 20px rgba(79, 195, 247, 0.3)' },
          '100%': { boxShadow: '0 0 40px rgba(206, 147, 216, 0.5)' },
        },
      },
    },
  },
  plugins: [],
}
