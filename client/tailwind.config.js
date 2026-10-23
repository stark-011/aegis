/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      boxShadow: {
        neon: '0 0 24px rgba(34, 197, 94, 0.16)',
      },
    },
  },
  plugins: [],
}
