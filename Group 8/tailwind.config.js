/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
      './templates/**/*.html',
      './accounts/templates/**/*.html',
      './d_staffs/templates/**/*.html',
      './*/templates/**/*.html',
  ],
  theme: {
    extend: {
      colors: {
        'brand': {
          DEFAULT:'#5E79FF',
          // DEFAULT:'#FA2A3C',
          hv:'#7189FF',
          light:'#9CADFF',
          100:'#DEE3FF',
        },
      }
    },
  },
  plugins: [
    // require('@tailwindcss/forms'),
  ],
}
