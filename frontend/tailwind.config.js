module.exports = {
  purge: ['./src/**/*.{js,jsx,ts,tsx}', './public/index.html'],
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {},
  },
  variants: {
    extend: {
       filter: ['group-hover'],
       grayscale: ['group-hover'],
       blur: ['group-hover'],
       brightness: ['group-hover'],
    },
  },
  plugins: [],
}
