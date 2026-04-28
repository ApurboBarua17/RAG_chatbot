/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ua: {
          red: '#AB0520',
          'red-dark': '#8B0015',
        },
      },
    },
  },
  plugins: [],
}
