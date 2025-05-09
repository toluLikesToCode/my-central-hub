// tests/router.test.ts
import { Socket } from 'net';
import routerInstance from '../../src/core/router'; // Import the singleton router instance only
import logger from '../../src/utils/logger';
import { sendResponse } from '../../src/entities/sendResponse';
import type { IncomingRequest } from '../../src/entities/http';

jest.mock('../../src/entities/sendResponse');
jest.mock('../../src/utils/logger', () => {
  const mockLogger = {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
    child: jest.fn().mockReturnValue({
      info: jest.fn(),
      error: jest.fn(),
      warn: jest.fn(),
      debug: jest.fn(),
    }),
  };

  return {
    __esModule: true, // Handle ES module interop
    default: mockLogger,
    Logger: jest.fn().mockImplementation(() => mockLogger),
    ConsoleTransport: jest.fn(),
    FileTransport: jest.fn(),
    PrettyFormatter: jest.fn(),
    JsonFormatter: jest.fn(),
  };
});

describe('Router', () => {
  let socket: Socket;
  let req: IncomingRequest;
  let mockChildLogger;

  beforeEach(() => {
    // Reset singleton router state for isolation
    routerInstance.reset();

    socket = {
      write: jest.fn(),
      end: jest.fn(),
      destroy: jest.fn(),
      remoteAddress: '127.0.0.1', // Added for logging context
    } as unknown as Socket;

    req = {
      method: 'GET',
      path: '/test',
      query: {},
      httpVersion: 'HTTP/1.1',
      headers: {},
      headersMap: new Map(),
      url: new URL('http://localhost/test'),
      raw: '',
      ctx: {},
      invalid: false,
    };

    // Capture the mock child logger instance returned by mocked logger
    mockChildLogger = logger.child({}); // Get the mock child
    (logger.child as jest.Mock).mockClear(); // Clear the call to child()
    Object.values(mockChildLogger).forEach((m) => (m as jest.Mock).mockClear()); // Clear methods on the child mock
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('should run middleware in sequence', async () => {
    const order: string[] = [];
    routerInstance.use(async (_req, _sock, next) => {
      order.push('mw1');
      await next();
      order.push('mw1-after');
    });
    routerInstance.use(async (_req, _sock, next) => {
      order.push('mw2');
      await next();
      order.push('mw2-after');
    });

    routerInstance.any('/test', async () => {
      order.push('handler');
    });
    await routerInstance.handle(req, socket);

    expect(order).toEqual(['mw1', 'mw2', 'handler', 'mw2-after', 'mw1-after']);
    // Note: Check basic logging calls
    expect(mockChildLogger.info).toHaveBeenCalledWith(expect.stringContaining('Request started'));
    expect(mockChildLogger.debug).toHaveBeenCalledWith(
      expect.stringContaining('Executing middleware #0'),
    );
    expect(mockChildLogger.debug).toHaveBeenCalledWith(
      expect.stringContaining('Executing middleware #1'),
    );
    expect(mockChildLogger.debug).toHaveBeenCalledWith(
      expect.stringContaining('Executing route handler'),
    );
    expect(mockChildLogger.info).toHaveBeenCalledWith(
      expect.stringContaining('Request completed'),
      expect.objectContaining({ status: 200 }),
    );
  });

  test('should call matching handler for GET route', async () => {
    const handler = jest.fn();
    routerInstance.get('/test', handler);

    await routerInstance.handle(req, socket);
    expect(handler).toHaveBeenCalledWith(req, socket);
  });

  test('should respond with 404 (text) if no route matches (non-API)', async () => {
    req.path = '/unknown';
    req.url = new URL('http://localhost/unknown');
    await routerInstance.handle(req, socket);

    expect(sendResponse).toHaveBeenCalledWith(
      socket,
      404,
      { 'Content-Type': 'text/plain' },
      'Not Found',
    );
    expect(mockChildLogger.warn).toHaveBeenCalledWith('No matching route found');
    expect(mockChildLogger.warn).toHaveBeenCalledWith(
      expect.stringContaining('Request completed'),
      expect.objectContaining({ status: 404 }),
    );
  });

  test('should respond with 404 (JSON) if no route matches (API)', async () => {
    req.path = '/api/unknown';
    req.url = new URL('http://localhost/api/unknown');
    await routerInstance.handle(req, socket);

    expect(sendResponse).toHaveBeenCalledWith(
      socket,
      404,
      { 'Content-Type': 'application/json' },
      JSON.stringify({ error: 'Not Found' }),
    );
    expect(mockChildLogger.warn).toHaveBeenCalledWith('No matching route found');
    expect(mockChildLogger.warn).toHaveBeenCalledWith(
      expect.stringContaining('Request completed'),
      expect.objectContaining({ status: 404 }),
    );
  });

  test('should respond with 405 (text) if method does not match (non-API)', async () => {
    req.method = 'POST';
    routerInstance.get('/test', jest.fn());
    await routerInstance.handle(req, socket);

    expect(sendResponse).toHaveBeenCalledWith(socket, 405, { Allow: 'GET' }, 'Method Not Allowed');
    expect(mockChildLogger.warn).toHaveBeenCalledWith(
      'Method not allowed',
      expect.objectContaining({ allowedMethods: 'GET' }),
    );
    expect(mockChildLogger.warn).toHaveBeenCalledWith(
      expect.stringContaining('Request completed'),
      expect.objectContaining({ status: 405 }),
    );
  });

  test('should respond with 405 (JSON) if method does not match (API)', async () => {
    req.method = 'POST';
    req.path = '/api/test';
    req.url = new URL('http://localhost/api/test');
    routerInstance.get('/api/test', jest.fn());
    await routerInstance.handle(req, socket);

    expect(sendResponse).toHaveBeenCalledWith(
      socket,
      405,
      { 'Content-Type': 'application/json', Allow: 'GET' },
      JSON.stringify({ error: 'Method Not Allowed' }),
    );
    expect(mockChildLogger.warn).toHaveBeenCalledWith(
      'Method not allowed',
      expect.objectContaining({ allowedMethods: 'GET' }),
    );
    expect(mockChildLogger.warn).toHaveBeenCalledWith(
      expect.stringContaining('Request completed'),
      expect.objectContaining({ status: 405 }),
    );
  });

  test('should respond with 500 (text) on handler error (non-API)', async () => {
    const testError = new Error('fail');
    routerInstance.get('/error', () => {
      throw testError;
    });

    req.path = '/error';
    req.url = new URL('http://localhost/error');
    await routerInstance.handle(req, socket);

    expect(sendResponse).toHaveBeenCalledWith(
      socket,
      500,
      { 'Content-Type': 'text/plain' },
      '500 Server Error',
    );
    expect(mockChildLogger.error).toHaveBeenCalledWith(
      'Handler error',
      expect.objectContaining({
        error: { message: 'fail', name: 'Error', stack: expect.any(String) },
      }),
    );
    expect(mockChildLogger.error).toHaveBeenCalledWith(
      expect.stringContaining('Request completed'),
      expect.objectContaining({ status: 500 }),
    );
  });

  test('should respond with 500 (JSON) on handler error (API)', async () => {
    const apiError = new Error('api fail');
    routerInstance.get('/api/error', () => {
      throw apiError;
    });

    req.path = '/api/error';
    req.url = new URL('http://localhost/api/error');
    await routerInstance.handle(req, socket);

    expect(sendResponse).toHaveBeenCalledWith(
      socket,
      500,
      { 'Content-Type': 'application/json' },
      JSON.stringify({ error: 'Internal Server Error' }),
    );
    // Updated to properly handle full error object with stack
    expect(mockChildLogger.error).toHaveBeenCalledWith(
      'Handler error',
      expect.objectContaining({
        error: expect.objectContaining({
          message: 'api fail',
          name: 'Error',
          stack: expect.any(String),
        }),
      }),
    );
    expect(mockChildLogger.error).toHaveBeenCalledWith(
      expect.stringContaining('Request completed'),
      expect.objectContaining({ status: 500 }),
    );
  });

  test('should respond to OPTIONS with Allow header', async () => {
    req.method = 'OPTIONS';
    req.path = '/anything';
    req.url = new URL('http://localhost/anything');
    await routerInstance.handle(req, socket);

    expect(sendResponse).toHaveBeenCalledWith(
      socket,
      204,
      expect.objectContaining({
        Allow: 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Content-Type': 'text/plain',
      }),
      'No Content',
    );
    expect(mockChildLogger.debug).toHaveBeenCalledWith('Responded to OPTIONS request');
    expect(mockChildLogger.info).toHaveBeenCalledWith(
      expect.stringContaining('Request completed'),
      expect.objectContaining({ status: 204 }),
    );
  });

  test('should route to /api/files/cache endpoint', async () => {
    const cacheHandler = jest.fn();
    routerInstance.get('/api/files/cache', cacheHandler);

    req.method = 'GET';
    req.path = '/api/files/cache';
    req.url = new URL('http://localhost/api/files/cache');
    await routerInstance.handle(req, socket);

    expect(cacheHandler).toHaveBeenCalledWith(req, socket);
  });
});
