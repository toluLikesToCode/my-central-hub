const js = require('@eslint/js');
const tseslint = require('typescript-eslint');
const prettier = require('eslint-plugin-prettier');
const markdown = require('eslint-plugin-markdown');

module.exports = [
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    ignores: ['dist/', 'node_modules/', 'coverage/', 'public/'],
    plugins: {
      prettier: prettier,
      markdown: markdown,
    },
    rules: {
      'prettier/prettier': 'error',
    },
  },
  {
    files: ['**/*.md'],
    plugins: {
      markdown: markdown,
    },
    processor: 'markdown/markdown',
  },
];
