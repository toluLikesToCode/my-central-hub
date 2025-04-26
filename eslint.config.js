const js = require("@eslint/js");
const tseslint = require("typescript-eslint");
const prettier = require("eslint-plugin-prettier");

module.exports = [
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    ignores: [
      "dist/",
      "node_modules/",
      "coverage/",
      "public/",
    ],
    plugins: {
      prettier: prettier,
    },
    rules: {
      "prettier/prettier": "error",  // <--- THIS ensures Prettier violations show as ESLint errors
      // Add any custom ESLint rules here
    },
  },
];
