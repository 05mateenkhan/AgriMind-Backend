/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        body: ['"Inter"', 'sans-serif'],
      },
      colors: {
        // Logo-extracted organic farm palette
        cream: '#F6F0D8',
        'cream-light': '#F3EACB',
        'cream-card': '#EFE3BF',
        'cream-highlight': '#FFF8E6',

        // Text (now black for maximum contrast on light creams/yellows)
        'text-dark': '#000000',
        'text-secondary': '#000000',
        'text-olive': '#000000',

        'green-primary': '#4A6F43',
        'green-secondary': '#7DAA6A',
        'green-light': '#CDE6C1',
        'green-button': '#9BCF88',

        'yellow-corn': '#F2C66D',
        'yellow-wheat': '#E3A857',

        'border-cream': '#E5D9B5',
      },
      backgroundImage: {
        'hero-gradient':
          'radial-gradient(circle at top left, rgba(205,230,193,0.4), transparent 45%), radial-gradient(circle at 20% 80%, rgba(243,234,203,0.7), transparent 40%), radial-gradient(circle at 80% 20%, rgba(155,207,136,0.35), transparent 35%)',
        'card-glass':
          'linear-gradient(135deg, rgba(239,227,191,0.98), rgba(246,240,216,0.98))',
      },
      boxShadow: {
        glow: '0 10px 30px rgba(155,207,136,0.35)',
        soft: '0 18px 45px rgba(44, 44, 44, 0.08)',
      },
      backgroundColor: (theme) => ({
        ...theme('colors'),
        DEFAULT: '#F6F0D8',
      }),
    },
  },
  plugins: [],
};

