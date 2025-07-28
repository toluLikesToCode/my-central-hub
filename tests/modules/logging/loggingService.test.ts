/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * Remote Logging Module Tests
 *
 * Tests for the remote logging functionality that handles incoming log entries
 * from remote clients using HttpProtocolHandler.
 */
import { Socket } from 'net';
import { IncomingRequest } from '../../../src/entities/http';
import {
  storeLogEntry,
  initializeDatabase,
  closeDatabase,
} from '../../../src/modules/logging/loggingService';
import { ingestLogs } from '../../../src/modules/logging/loggingController';
import fs from 'fs';
import path from 'path';
import sqlite3 from 'sqlite3';
import { open as sqliteOpen } from 'sqlite';

// Mock the sendWithContext function
jest.mock('../../../src/entities/sendResponse', () => ({
  sendWithContext: jest.fn((req, sock, status, headers, body) => {
    console.log('sendWithContext called with status:', status);
    console.log('Headers:', headers);
    console.log('Body:', body);
  }),
}));

// Import the mocked sendWithContext
import { sendWithContext } from '../../../src/entities/sendResponse';

beforeAll(() => {
  jest.doMock('../../../src/config/server.config', () => ({
    config: {
      dbPath: path.join(__dirname, '../../temp/test_logs.db'),
      adminKey: 'test-admin-key',
      logging: {
        logDir: path.join(__dirname, '../../temp'),
      },
    },
  }));
});

describe('Remote Logging Module', () => {
  let testDbPath: string;

  beforeAll(async () => {
    // Make sure the temp directory exists
    const tempDir = path.join(__dirname, '../../temp');
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }
    // Always use a temp DB for tests
    testDbPath = path.join(tempDir, 'test_logs.db');
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
    await initializeDatabase(testDbPath);
  });

  afterAll(async () => {
    await closeDatabase();
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  beforeEach(async () => {
    jest.clearAllMocks();
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
    await initializeDatabase(testDbPath);
  });

  describe('storeLogEntry', () => {
    test('should store a valid log entry with all fields', async () => {
      const entry = {
        level: 'info',
        message: { text: 'Test log message', extra: 123 },
        timestamp: new Date().toISOString(),
        context: 'TestContext',
        meta: { foo: 'bar' },
        url: 'http://localhost/test',
        correlationId: 'corr-123',
        consoleArgs: ['msg', { a: 1 }],
        userAgent: 'Jest Test Agent',
        ip: '1.2.3.4',
        extraField: 'should be in raw_entry',
      };
      const id = await storeLogEntry(entry);
      expect(id).toBeGreaterThan(0);
      // Optionally, query the DB directly to check columns (if you expose a getLogs or similar)
    });

    test('should store an entry with error and fallback fields', async () => {
      const entry = {
        level: 'error',
        message: 'Test error message',
        timestamp: new Date().toISOString(),
        error: {
          name: 'TestError',
          message: 'This is a test error',
          stack: 'at TestFunction (test.ts:10:20)',
          code: 'TEST_CODE',
        },
        contextName: 'TestContext',
        meta: { foo: 'bar' },
      };
      const id = await storeLogEntry(entry, '5.6.7.8', 'UA-Override');
      expect(id).toBeGreaterThan(0);
    });

    test('should store and serialize extra fields in raw_entry', async () => {
      const entry = {
        level: 'debug',
        message: 'Extra fields test',
        timestamp: new Date().toISOString(),
        context: 'ExtraContext',
        foo: 'bar',
        bar: 42,
        baz: { nested: true },
      };
      const id = await storeLogEntry(entry);
      expect(id).toBeGreaterThan(0);
    });

    test('should store a log entry with deeply nested meta and clientLog', async () => {
      const entry = {
        level: 'debug',
        message: 'Received client-side log',
        contextName: 'gallery-server',
        meta: {
          clientLog: {
            level: 'info',
            message: 'Modal System Initialized.',
            consoleArgs: [
              'Modal System Initialized.',
              { correlationId: 'client-1747485125120-fb5234513dc058' },
              'http://192.168.1.145:3456',
            ],
            url: 'http://localhost:3456/',
            userAgent:
              'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:138.0) Gecko/20100101 Firefox/138.0',
            timestamp: new Date().toISOString(),
            context: 'ModalManager',
          },
          ip: '192.168.1.145',
          userAgent:
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:138.0) Gecko/20100101 Firefox/138.0',
          timestamp: new Date().toISOString(),
        },
        timestamp: new Date().toISOString(),
      };
      const id = await storeLogEntry(entry);
      expect(id).toBeGreaterThan(0);
    });

    test('should store a log entry with all possible fields and arbitrary extras', async () => {
      const entry = {
        level: 'info',
        message: 'Processing triad with center: RosybrownExtralargeSalmon.mp4 (group 190)',
        contextName: 'groupings',
        meta: {},
        timestamp: new Date().toISOString(),
        url: 'http://localhost:3456/',
        correlationId: 'client-1747516971291-c608fda5e4c698',
        consoleArgs: [
          'Flushed audioStates to localStorage.',
          { operation: 'flushAudioStates' },
          'http://192.168.1.145:3456',
        ],
        userAgent:
          'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:138.0) Gecko/20100101 Firefox/138.0',
        ip: '192.168.1.145',
        extra1: 123,
        extra2: { foo: 'bar' },
      };
      const id = await storeLogEntry(entry);
      expect(id).toBeGreaterThan(0);
    });

    test('should extract and index fields from deeply nested log entry', async () => {
      const now = new Date().toISOString();
      const entry = {
        // Provide dummy top-level required fields (should be ignored by recursive extraction)
        level: 'dummy',
        message: 'dummy',
        timestamp: '2000-01-01T00:00:00.000Z',
        meta: {
          foo: {
            bar: {
              level: 'warn',
              message: 'Deeply nested warning',
              contextName: 'DeepContext',
              timestamp: now,
              url: 'http://deep.example.com',
              correlationId: 'deep-corr-id',
              consoleArgs: ['deep', { deep: true }],
              userAgent: 'DeepTestAgent/1.0',
              ip: '9.8.7.6',
              error: {
                name: 'DeepError',
                message: 'Something went deep wrong',
                stack: 'at DeepFunction (deep.ts:1:1)',
                code: 'DEEP_FAIL',
              },
            },
          },
        },
      };
      const id = await storeLogEntry(entry);
      expect(id).toBeGreaterThan(0);
      // Query the DB directly to verify extraction
      const db = await sqliteOpen({ filename: testDbPath, driver: sqlite3.Database });
      const row = await db.get('SELECT * FROM logs WHERE id = ?', id);
      expect(row.level).toBe('warn');
      expect(row.message).toBe('Deeply nested warning');
      expect(row.contextName).toBe('DeepContext');
      expect(row.timestamp).toBe(now);
      expect(row.url).toBe('http://deep.example.com');
      expect(row.correlationId).toBe('deep-corr-id');
      expect(JSON.parse(row.consoleArgs)).toEqual(['deep', { deep: true }]);
      expect(row.user_agent).toBe('DeepTestAgent/1.0');
      expect(row.client_ip).toBe('9.8.7.6');
      expect(row.error_name).toBe('DeepError');
      expect(row.error_message).toBe('Something went deep wrong');
      expect(row.error_stack).toBe('at DeepFunction (deep.ts:1:1)');
      const errorDetails = JSON.parse(row.error_details);
      expect(errorDetails.code).toBe('DEEP_FAIL');
      await db.close();
    });

    test('should store a log entry with message as array, number, boolean, or null', async () => {
      const entries = [
        { level: 'info', message: [1, 2, 3], timestamp: new Date().toISOString() },
        { level: 'info', message: 42, timestamp: new Date().toISOString() },
        { level: 'info', message: true, timestamp: new Date().toISOString() },
        { level: 'info', message: null, timestamp: new Date().toISOString() },
      ];
      for (const entry of entries) {
        const id = await storeLogEntry(entry);
        expect(id).toBeGreaterThan(0);
        const db = await sqliteOpen({ filename: testDbPath, driver: sqlite3.Database });
        const row = await db.get('SELECT * FROM logs WHERE id = ?', id);
        expect(row.level).toBe('info');
        expect(row.timestamp).toBe(entry.timestamp);
        // Message is stringified for storage
        expect(row.message).toBe(JSON.stringify(entry.message));
        await db.close();
      }
    });

    test('should store a log entry with arrays/objects in unexpected places', async () => {
      const entry = {
        level: 'warn',
        message: 'Array in contextName',
        contextName: [1, 2, 3],
        meta: { foo: { bar: [4, 5, 6] } },
        timestamp: new Date().toISOString(),
      };
      const id = await storeLogEntry(entry as any);
      expect(id).toBeGreaterThan(0);
      const db = await sqliteOpen({ filename: testDbPath, driver: sqlite3.Database });
      const row = await db.get('SELECT * FROM logs WHERE id = ?', id);
      // contextName should be stringified or ignored, but not crash
      expect(row.level).toBe('warn');
      expect(row.timestamp).toBe(entry.timestamp);
      await db.close();
    });

    test('should handle log entry with circular references gracefully', async () => {
      const entry: any = {
        level: 'error',
        message: 'Circular ref',
        timestamp: new Date().toISOString(),
      };
      entry.self = entry;
      let errorCaught = false;
      try {
        await storeLogEntry(entry);
      } catch (err) {
        errorCaught = true;
        expect(String(err)).toMatch(/circular/i);
      }
      expect(errorCaught).toBe(true);
    });

    test('should store a log entry with a very large object', async () => {
      const bigObj = {};
      for (let i = 0; i < 1000; i++) bigObj['key' + i] = 'x'.repeat(100);
      const entry = {
        level: 'info',
        message: bigObj,
        timestamp: new Date().toISOString(),
      };
      const id = await storeLogEntry(entry);
      expect(id).toBeGreaterThan(0);
      const db = await sqliteOpen({ filename: testDbPath, driver: sqlite3.Database });
      const row = await db.get('SELECT * FROM logs WHERE id = ?', id);
      expect(row.level).toBe('info');
      expect(row.timestamp).toBe(entry.timestamp);
      expect(row.message.length).toBeGreaterThan(1000); // Should be stringified
      await db.close();
    });

    test('should store a log entry with only extra fields and no standard fields', async () => {
      const entry = { foo: 'bar', baz: 123, qux: [1, 2, 3] };
      let errorCaught = false;
      try {
        await storeLogEntry(entry as any);
      } catch (err) {
        errorCaught = true;
        expect(String(err)).toMatch(/level|timestamp|message/i);
      }
      expect(errorCaught).toBe(true);
    });

    test('should store a log entry with malformed/non-ISO timestamp', async () => {
      const entry = {
        level: 'info',
        message: 'Bad timestamp',
        timestamp: 'not-a-date',
      };
      const id = await storeLogEntry(entry);
      expect(id).toBeGreaterThan(0);
      const db = await sqliteOpen({ filename: testDbPath, driver: sqlite3.Database });
      const row = await db.get('SELECT * FROM logs WHERE id = ?', id);
      expect(row.timestamp).toBe('not-a-date');
      await db.close();
    });

    test('should store a log entry with deeply nested arrays and mixed types', async () => {
      const entry = {
        level: 'debug',
        message: 'Deep array',
        meta: {
          arr: [1, { a: [2, 3, { b: [4, 5, [6, { c: 'end' }]] }] }, 'str', null],
        },
        timestamp: new Date().toISOString(),
      };
      const id = await storeLogEntry(entry);
      expect(id).toBeGreaterThan(0);
      const db = await sqliteOpen({ filename: testDbPath, driver: sqlite3.Database });
      const row = await db.get('SELECT * FROM logs WHERE id = ?', id);
      expect(row.level).toBe('debug');
      expect(row.timestamp).toBe(entry.timestamp);
      await db.close();
    });
  });

  describe('ingestLogs', () => {
    // Helper to create a mock request
    function createMockRequest(
      body: unknown,
      headers: Record<string, string> = {},
    ): IncomingRequest {
      return {
        method: 'POST',
        url: '/api/logs/ingest',
        headers: headers,
        body: Buffer.from(JSON.stringify(body)),
        path: '/api/logs/ingest',
        query: {},
      } as unknown as IncomingRequest;
    }

    // Helper to create a mock socket
    function createMockSocket(): Socket {
      return {
        remoteAddress: '127.0.0.1',
        destroyed: false,
      } as unknown as Socket;
    }

    test('should accept a valid batch of log entries', async () => {
      const mockReq = createMockRequest([
        {
          level: 'info',
          message: 'Server started',
          timestamp: new Date().toISOString(),
        },
        {
          level: 'error',
          message: 'Connection failed',
          contextName: 'DatabaseService',
          timestamp: new Date().toISOString(),
          error: {
            name: 'ConnectionError',
            message: 'Failed to connect to database',
            stack: 'Error: Failed to connect to database\n    at connectDB (/app/db.ts:45:10)',
          },
        },
      ]);

      const mockSock = createMockSocket();

      await ingestLogs(mockReq, mockSock);

      expect(sendWithContext).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        200,
        expect.objectContaining({
          'Content-Type': 'application/json',
        }),
        expect.stringContaining('"status":"ok"'),
      );
    });

    test('should reject non-array payloads', async () => {
      const mockReq = createMockRequest({ message: 'Not an array' });
      const mockSock = createMockSocket();

      await ingestLogs(mockReq, mockSock);

      expect(sendWithContext).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        400,
        expect.objectContaining({
          'Content-Type': 'application/json',
        }),
        expect.stringContaining('Invalid request format'),
      );
    });

    test('should reject invalid log entries', async () => {
      const mockReq = createMockRequest([
        {
          // Missing required fields
          contextName: 'InvalidLog',
        },
      ]);
      const mockSock = createMockSocket();

      await ingestLogs(mockReq, mockSock);

      expect(sendWithContext).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        400,
        expect.objectContaining({
          'Content-Type': 'application/json',
        }),
        expect.stringContaining('All log entries were invalid'),
      );
    });

    test('should handle partial success with some invalid entries', async () => {
      const mockReq = createMockRequest([
        {
          level: 'info',
          message: 'Valid entry',
          timestamp: new Date().toISOString(),
        },
        {
          // Invalid entry (missing required fields)
          contextName: 'InvalidLog',
        },
      ]);
      const mockSock = createMockSocket();

      await ingestLogs(mockReq, mockSock);

      expect(sendWithContext).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        207,
        expect.objectContaining({
          'Content-Type': 'application/json',
        }),
        expect.stringContaining('"status":"partial"'),
      );
    });

    test.skip('should authenticate with valid Bearer token', async () => {
      const mockReq = createMockRequest(
        [
          {
            level: 'info',
            message: 'Authenticated log',
            timestamp: new Date().toISOString(),
          },
        ],
        {
          authorization: 'Bearer test-admin-key',
        },
      );
      const mockSock = createMockSocket();

      await ingestLogs(mockReq, mockSock);

      expect(sendWithContext).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        200,
        expect.objectContaining({
          'Content-Type': 'application/json',
        }),
        expect.stringContaining('"status":"ok"'),
      );
    });

    test.skip('should authenticate with valid API key', async () => {
      const mockReq = createMockRequest(
        [
          {
            level: 'info',
            message: 'API key authenticated log',
            timestamp: new Date().toISOString(),
          },
        ],
        {
          authorization: 'ApiKey test-admin-key',
        },
      );
      const mockSock = createMockSocket();

      await ingestLogs(mockReq, mockSock);

      expect(sendWithContext).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        200,
        expect.objectContaining({
          'Content-Type': 'application/json',
        }),
        expect.stringContaining('"status":"ok"'),
      );
    });

    test('should reject with invalid authentication', async () => {
      const mockReq = createMockRequest(
        [
          {
            level: 'info',
            message: 'Unauthenticated log',
            timestamp: new Date().toISOString(),
          },
        ],
        {
          authorization: 'Bearer invalid-token',
        },
      );
      const mockSock = createMockSocket();

      await ingestLogs(mockReq, mockSock);

      expect(sendWithContext).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        401,
        expect.objectContaining({
          'Content-Type': 'application/json',
        }),
        expect.stringContaining('"error":"Unauthorized"'),
      );
    });
  });
});
