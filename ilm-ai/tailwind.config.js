/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/App.jsx", // Add this for a very direct test
    "./src/**/*.{js,ts,jsx,tsx}", // This line ensures Tailwind scans your React components
  ],
  theme: {
    extend: {
      fontFamily: {
        // Optional: Add 'inter' if you want to use it as in your code
        inter: ["Inter", "sans-serif"],
      },
    },
  },
  plugins: [
    //require('@tailwindcss/typography'), // Add this if you use the 'prose' class
  ],
  darkMode: "class", // Or 'media' if you prefer system preference for dark mode
};
