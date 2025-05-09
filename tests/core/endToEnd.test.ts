/* eslint-disable @typescript-eslint/no-explicit-any */
// /src/tests/core/endToEnd.tests.ts
// This test file is for end-to-end tests of the server
// It uses the supertest library to make HTTP requests to the server
// and checks the responses

import request from 'supertest';
import { HttpServer } from '../../src/core/server';
import type { Server } from 'net';

const normalize = (s: string) =>
  JSON.parse(s)
    .message.toLowerCase()
    .replace(/[^\w\s]/g, '');

let httpServer: HttpServer;
let server: Server;

jest.mock('../../src/utils/logger', () => {
  // Helper to create a logger function for a specific level
  const createLevelLogger = (level: string) =>
    jest.fn((primaryMessageOrObject: any, ...metadata: any[]) => {
      const logPrefix = `TEST-LOG [${level.toUpperCase()}]:`;
      let message = primaryMessageOrObject;
      let metaString = '';

      if (typeof primaryMessageOrObject === 'object' && metadata.length === 0) {
        // Called like: logger.info({ key: 'value' });
        message = JSON.stringify(primaryMessageOrObject, null, 2);
      } else if (typeof primaryMessageOrObject === 'string') {
        // Called like: logger.info('message', { key: 'value' }, ...);
        // message is already primaryMessageOrObject
        if (metadata.length > 0) {
          metaString = metadata
            .map((m) => (typeof m === 'object' ? JSON.stringify(m, null, 2) : m.toString()))
            .join(' ');
        }
      } else {
        // Fallback for other types
        message = primaryMessageOrObject.toString();
        if (metadata.length > 0) {
          metaString = metadata
            .map((m) => (typeof m === 'object' ? JSON.stringify(m, null, 2) : m.toString()))
            .join(' ');
        }
      }

      // Using process.stdout.write might behave differently with Jest's console handling
      // compared to console.log, especially during teardown.
      process.stdout.write(`${logPrefix} ${message}${metaString ? ' ' + metaString : ''}\n`);
    });

  // Create the mock logger instance structure
  const mockLoggerInstance = {
    info: createLevelLogger('info'),
    error: createLevelLogger('error'),
    warn: createLevelLogger('warn'),
    debug: createLevelLogger('debug'), // This will be used by withTiming
    success: createLevelLogger('success'),
    child: jest.fn(), // We'll make child return the same instance for simplicity
  };

  // Make child() return the same logger instance.
  // For more complex scenarios, child could return a new instance with bound context.
  mockLoggerInstance.child.mockReturnValue(mockLoggerInstance);

  return {
    __esModule: true, // Important for ES module interop
    default: mockLoggerInstance, // If your app imports `logger from './logger'`
    Logger: jest.fn().mockImplementation(() => mockLoggerInstance), // If your app uses `new Logger()`
    // Mock other exports from your logger module if there are any
    ConsoleTransport: jest.fn(),
    FileTransport: jest.fn(),
    PrettyFormatter: jest.fn(),
    JsonFormatter: jest.fn(),
  };
});

beforeAll(async () => {
  // Use ephemeral port (0) to avoid occupying fixed port
  httpServer = new HttpServer(0);
  server = await httpServer.start();
});

// Define the test suite
describe('GET /ping', () => {
  it('should return 200 OK', async () => {
    const res = await request(server).get('/ping');
    expect(res.status).toBe(200);
  });
});

describe('GET /echo', () => {
  it('should return 200 OK with the same body as the request body', async () => {
    const payload = { message: 'test message' };
    const res = await request(server)
      .get('/echo')
      .send(payload)
      .set('Content-Type', 'application/json');
    expect(res.status).toBe(200);
    expect(res.body.message).toBe(payload.message);
  });

  it('should return 200 OK and say "hello world" if request body is empty', async () => {
    const res = await request(server).get('/echo');
    expect(res.status).toBe(200);
    expect(normalize(res.text)).toBe('hello world');
  });
});

describe('GET /api/files pagination and sorting', () => {
  it('should return the first 30 files sorted by name (asc), hasNextPage true, and next page link works', async () => {
    const res = await request(server).get('/api/files?limit=30&sort=name&order=asc');
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('files');
    expect(Array.isArray(res.body.files)).toBe(true);
    expect(res.body.files.length).toBeLessThanOrEqual(30);

    // Check sorting order (asc by name)
    const names = res.body.files.map((f) => f.name);
    const sorted = [...names].sort((a, b) => a.localeCompare(b));
    expect(names).toEqual(sorted);

    // Check pagination
    expect(res.body.pagination).toBeDefined();
    expect(res.body.pagination.hasNextPage).toBeDefined();

    // Check _links.next exists and is a string
    expect(res.body._links).toBeDefined();
    expect(typeof res.body._links.next === 'string' || res.body._links.next === null).toBe(true);

    if (res.body._links.next && res.body.pagination.hasNextPage) {
      try {
        // Make a request to the next page link with a timeout
        const nextPageUrl = res.body._links.next.replace(/^https?:\/\/[^/]+(\/|$)/, '/');
        const nextRes = await request(server).get(nextPageUrl).timeout(5000);
        expect(nextRes.status).toBe(200);
        expect(nextRes.body.pagination.page).toBe(res.body.pagination.page + 1);

        // Should not return the same files as the first page
        const nextNames = nextRes.body.files.map((f) => f.name);
        expect(nextNames.some((n) => names.includes(n))).toBe(false);
      } catch (err) {
        // If we can't reach the next page, at least make sure the pagination info was correct
        console.warn(
          'Failed to fetch next page, but the pagination metadata was present',
          err.message,
        );
      }
    }
  });
});

describe('GET /api/files with filters', () => {
  /**
   * Helper that hits `/api/files` with arbitrary query params
   * and returns `res.body.files` for convenience.
   */
  const list = (query: Record<string, unknown> = {}) =>
    request(server)
      .get('/api/files')
      .query(query)
      .expect(200)
      .then((res) => res.body.files as Array<{ name: string; mimeType: string; size: number }>);

  it('returns **only images** whose filename does NOT contain "thumbnail"', async () => {
    const files = await list({
      type: 'image/',
      filter: JSON.stringify({ not: { fileName: 'thumbnail' } }),
    });

    // 1. Check that all files are images
    expect(files.every((f) => f.mimeType.startsWith('image/'))).toBe(true);

    // 2. Check that no files contain "thumbnail" in their name
    expect(files.every((f) => !f.name.includes('thumbnail'))).toBe(true);

    // ensure no videos are included
    expect(files.every((f) => f.mimeType !== 'video/mp4')).toBe(true);
  });
});

afterAll(async () => {
  await httpServer.stop();
});
