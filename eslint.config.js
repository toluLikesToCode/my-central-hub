const js = require('@eslint/js');
const tseslint = require('typescript-eslint');
const prettier = require('eslint-plugin-prettier');
const markdown = require('eslint-plugin-markdown');
const importPlugin = require('eslint-plugin-import');

module.exports = [
  {
    ignores: ['**/*.md'],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    ignores: ['dist/', 'node_modules/', 'coverage/', 'public/'],
    settings: {
      'import/resolver': {
        typescript: {
          project: './tsconfig.json',
        },
      },
    },
    plugins: {
      prettier: prettier,
      markdown: markdown,
      import: importPlugin,
    },
    rules: {
      'prettier/prettier': 'error',
      'import/no-unresolved': ['error', { commonjs: true, amd: true }],
    },
  },
  {
    files: ['**/*.md'],
  },
];
