/* eslint-disable no-undef */
/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/tests/**/*.test.ts', '**/stress/**/*.test.ts'], // <--- IMPORTANT
  moduleFileExtensions: ['ts', 'js', 'json', 'node'],
  modulePathIgnorePatterns: ['<rootDir>/output/'],
  setupFilesAfterEnv: ['jest-extended/all', '<rootDir>/tests/setup.jest.ts'],
  transform: {
    '^.+\\.(ts|tsx)$': [
      'ts-jest',
      {
        tsconfig: 'tsconfig.json',
      },
    ],
  },
  transformIgnorePatterns: [
    '/node_modules/(?!(boxen|chalk|ansi-styles|supports-color|has-flag|color-convert|color-name)/)',
  ],
};
