/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Page background. Deep warm brown-black, not pure black.
        base: {
          900: "#0e0908",
          800: "#171110",
          700: "#1f1817",
          600: "#2a2018",
        },
        // Text. Warm cream-white, not pure white.
        cream: {
          100: "#f4ebd9",
          300: "#a89684",
          500: "#6b5e51",
        },
        // Saffron. Glow against dark.
        saffron: {
          400: "#e8a04c",
          600: "#c8741b",
          700: "#a45d14",
        },
        // Sentiment. Warmer than default Tailwind.
        sage: {
          500: "#7ab87a",
        },
        terracotta: {
          500: "#d97757",
        },
      },
      fontFamily: {
        display: ['"Cormorant Garamond"', "ui-serif", "Georgia", "serif"],
        serif: ["Lora", "ui-serif", "Georgia", "serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
