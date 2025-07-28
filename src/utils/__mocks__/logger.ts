// tests/__mocks__/logger.ts
/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */

import { jest } from '@jest/globals'; // Use import for Jest

// --- Mock Functions ---
// Create distinct mock functions for clarity, though Jest resets them anyway.
const mockInfo = jest.fn();
const mockError = jest.fn();
const mockWarn = jest.fn();
const mockDebug = jest.fn();
const mockHttp = jest.fn();
const mockVerbose = jest.fn();
const mockSilly = jest.fn();
const mockSuccess = jest.fn(); // Alias for info

// --- Mock Child Logger ---
// This mock child object recursively returns itself for the `child` method,
// allowing for infinite chaining like logger.child({...}).child({...}).info(...)
const mockChildLoggerInstance = {
  info: jest.fn(),
  error: jest.fn(),
  warn: jest.fn(),
  debug: jest.fn(),
  http: jest.fn(),
  verbose: jest.fn(),
  silly: jest.fn(),
  success: jest.fn(),
  // Crucially, child returns itself
  child: jest.fn(() => mockChildLoggerInstance),
};

// --- Default Export Mock ---
// This is the object that will be imported via `import logger from '...'`
const mockLogger = {
  info: mockInfo,
  error: mockError,
  warn: mockWarn,
  debug: mockDebug,
  http: mockHttp,
  verbose: mockVerbose,
  silly: mockSilly,
  success: mockSuccess, // Usually maps to info
  child: jest.fn(() => mockChildLoggerInstance), // The main child call
};

// --- Named Exports Mocks ---

// Mock the Logger class
// When `new Logger()` is called in tests, it returns the singleton mockLogger.
export class Logger {
  constructor(options: any = {}) {
    // Optionally log constructor calls if needed for debugging tests
    // console.log('[Mock Logger] Constructor called with:', options);
    return mockLogger; // Return the singleton mock instance
  }

  // Ensure the instance methods are also mocked if they are ever called directly
  // on an instance (though returning the singleton is the common pattern).
  info = mockInfo;
  error = mockError;
  warn = mockWarn;
  debug = mockDebug;
  http = mockHttp;
  verbose = mockVerbose;
  silly = mockSilly;
  success = mockSuccess;
  child = jest.fn(() => mockChildLoggerInstance);
}

// Mock Formatter classes
export class JsonFormatter {
  constructor(options: any = {}) {}
  format = jest.fn().mockReturnValue('{"mocked": "json"}');
}

export class PrettyFormatter {
  constructor(options: any = {}) {}
  format = jest.fn().mockReturnValue('mocked pretty format');
}

// Mock Transport classes
export class ConsoleTransport {
  public formatter: any; // Add formatter property
  public level?: string; // Add level property
  constructor(options: any = {}) {
    this.formatter = options.formatter || new PrettyFormatter(); // Match interface
    this.level = options.level;
  }
  log = jest.fn();
  close = jest.fn();
}

export class FileTransport {
  public formatter: any; // Add formatter property
  public level?: string; // Add level property
  constructor(options: any = {}) {
    this.formatter = options.formatter || new JsonFormatter(); // Match interface
    this.level = options.level;
  }
  log = jest.fn();
  close = jest.fn();
}

// Mock standardLevels if it's exported and used
// (Assuming it's exported based on src/utils/logger.ts content)
export const standardLevels = {
  error: 0,
  warn: 1,
  info: 2,
  http: 3,
  verbose: 4,
  debug: 5,
  silly: 6,
};

// Mock the sharedAppLogTransport if it's exported and potentially used directly
// Give it a mock formatter and level to satisfy the Transport interface
export const sharedAppLogTransport = new FileTransport({
  filename: 'mock_app.log',
  formatter: new JsonFormatter(),
  level: 'silly',
});

// --- Default Export ---
export default mockLogger; // Use ES6 default export
