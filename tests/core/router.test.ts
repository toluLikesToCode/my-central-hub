/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck
import { Socket } from 'net';
import { createRouter } from '../../src/core/router';
import type { IncomingRequest } from '../../src/entities/http';
import { sendResponse } from '../../src/entities/sendResponse';
import * as sendResponseModule from '../../src/entities/sendResponse';
import { corsMiddleware, handlePreflightRequest } from '../../src/core/middlewares/cors';

jest.mock('../../src/entities/sendResponse', () => {
  const actual = jest.requireActual('../../src/entities/sendResponse');
  return {
    ...actual,
    // keep sendWithContext real, only mock sendResponse
    sendResponse: jest.fn((...args) => {
      console.log('Mocked sendResponse called with:', {
        method: args[0]?.method || 'unknown',
        status: args[1] || 'unknown',
        headers: args[2] || {},
      });
      return actual.sendResponse(...args);
    }),
  };
});

describe('Router', () => {
  let router;
  let socket: Socket;
  let req: IncomingRequest;

  beforeEach(() => {
    // Create a new router instance for each test and reset it to remove default middlewares
    router = createRouter();
    router.reset(); // Clear all middlewares and routes

    router.use((req, sock, next) => {
      console.log('Before CORS middleware, method:', req.method);
      return next();
    });

    router.use(corsMiddleware);

    router.use((req, sock, next) => {
      console.log('After CORS middleware, method:', req.method);
      return next();
    });

    // Add direct OPTIONS handler for testing
    router.options('/api/test', (req, sock) => {
      console.log('OPTIONS handler called directly');
      handlePreflightRequest(req, sock);
    });

    socket = {
      write: jest.fn(),
      end: jest.fn(),
      destroy: jest.fn(),
      remoteAddress: '127.0.0.1',
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
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Request Context', () => {
    test('should preserve and pass request context through middleware chain', async () => {
      router.use(async (req, _sock, next) => {
        req.ctx.fromMiddleware1 = 'value1';
        await next();
      });

      router.use(async (req, _sock, next) => {
        req.ctx.fromMiddleware2 = 'value2';
        await next();
      });

      const handler = jest.fn();
      router.get('/test', handler);

      await router.handle(req, socket);

      expect(handler).toHaveBeenCalledWith(
        expect.objectContaining({
          ctx: expect.objectContaining({
            fromMiddleware1: 'value1',
            fromMiddleware2: 'value2',
          }),
        }),
        socket,
      );
    });

    test('should maintain context on error in middleware', async () => {
      router.use(async (req, _sock, next) => {
        req.ctx.beforeError = 'preserved';
        await next();
      });

      router.use(async () => {
        throw new Error('middleware error');
      });

      await router.handle(req, socket);
      expect(req.ctx.beforeError).toBe('preserved');
      expect(sendResponse).toHaveBeenCalledWith(
        socket,
        500,
        expect.any(Object),
        expect.any(String),
      );
    });
  });

  describe('Route Matching', () => {
    test('should match routes with path parameters', async () => {
      const handler = jest.fn();
      router.get('/users/:id/posts/:postId', handler);

      req.path = '/users/123/posts/456';
      req.url = new URL('http://localhost/users/123/posts/456');
      await router.handle(req, socket);

      expect(handler).toHaveBeenCalledTimes(1);
      // Params are attached to req.ctx.params
      const calledReq = handler.mock.calls[0][0];
      expect(calledReq.ctx.params).toEqual({ id: '123', postId: '456' });
      expect(calledReq.path).toBe('/users/123/posts/456');
      expect(calledReq.method).toBe('GET');
    });

    test('should match wildcard routes', async () => {
      const handler = jest.fn();
      router.get('/files/*', handler);

      req.path = '/files/images/photo.jpg';
      req.url = new URL('http://localhost/files/images/photo.jpg');
      await router.handle(req, socket);

      expect(handler).toHaveBeenCalledTimes(1);
      const calledReq = handler.mock.calls[0][0];
      expect(calledReq.ctx.params).toEqual({ '*': 'images/photo.jpg' });
    });

    test('should match exact routes before wildcards', async () => {
      const exactHandler = jest.fn();
      const wildcardHandler = jest.fn();

      router.get('/files/special', exactHandler);
      router.get('/files/*', wildcardHandler);

      req.path = '/files/special';
      req.url = new URL('http://localhost/files/special');
      await router.handle(req, socket);

      expect(exactHandler).toHaveBeenCalledTimes(1);
      expect(wildcardHandler).not.toHaveBeenCalled();
    });
  });

  describe('HTTP Methods', () => {
    test.each([
      ['get', 'GET'],
      ['post', 'POST'],
      ['put', 'PUT'],
      ['del', 'DELETE'],
      ['head', 'HEAD'],
    ])('should register and match %s routes', async (method, httpMethod) => {
      const handler = jest.fn();
      router[method]('/test', handler);

      req.method = httpMethod;
      await router.handle(req, socket);

      expect(handler).toHaveBeenCalledTimes(1);
    });

    test('should handle HEAD requests for GET routes', async () => {
      const handler = jest.fn();
      router.get('/test', handler);

      req.method = 'HEAD';
      await router.handle(req, socket);

      expect(sendResponse).toHaveBeenCalledWith(
        expect.any(Object),
        200,
        expect.any(Object),
        undefined, // HEAD responses should have no body
      );
    });
  });

  describe('Middleware Execution', () => {
    test('should execute middleware in correct order', async () => {
      const order: string[] = [];
      const createMiddleware = (name: string) => {
        return async (_req: IncomingRequest, _sock: Socket, next: () => Promise<void>) => {
          order.push(`${name}-before`);
          await next();
          order.push(`${name}-after`);
        };
      };

      router.use(createMiddleware('first'));
      router.use(createMiddleware('second'));
      router.get('/test', () => {
        order.push('handler');
      });

      await router.handle(req, socket);

      expect(order).toEqual([
        'first-before',
        'second-before',
        'second-after',
        'first-after',
        'handler',
      ]);
    });

    test('should stop middleware chain on error', async () => {
      const order: string[] = [];

      router.use(async (_req, _sock, next) => {
        order.push('first');
        await next();
        // 'first-after' should NOT run after error
      });

      router.use(async () => {
        order.push('error');
        throw new Error('middleware error');
      });

      router.use(async (_req, _sock, next) => {
        order.push('never');
        await next();
      });

      await router.handle(req, socket);

      expect(order).toEqual(['first', 'error']);
      expect(sendResponse).toHaveBeenCalledWith(
        socket,
        500,
        expect.any(Object),
        expect.any(String),
      );
    });
  });

  describe('Error Handling', () => {
    test('should handle sync errors in handlers', async () => {
      router.get('/error', () => {
        throw new Error('sync error');
      });

      req.path = '/error';
      req.url = new URL('http://localhost/error');
      await router.handle(req, socket);

      const [sockArg, statusArg, headersArg, bodyArg] = sendResponse.mock.calls[0];
      expect(sockArg).toBe(socket);
      expect(statusArg).toBe(500);
      expect(headersArg['Content-Type']).toBe('text/plain');
      expect(bodyArg).toBe('500 Server Error');
      expect(sendResponse).toHaveBeenCalledTimes(1);
    });

    test('should handle async errors in handlers', async () => {
      router.get('/error', async () => {
        throw new Error('async error');
      });

      req.path = '/error';
      req.url = new URL('http://localhost/error');
      await router.handle(req, socket);

      const [sockArg, statusArg, headersArg, bodyArg] = sendResponse.mock.calls[0];
      expect(sockArg).toBe(socket);
      expect(statusArg).toBe(500);
      expect(headersArg['Content-Type']).toBe('text/plain');
      expect(bodyArg).toBe('500 Server Error');
      expect(sendResponse).toHaveBeenCalledTimes(1);
    });

    test('should format API errors as JSON', async () => {
      router.get('/api/error', () => {
        throw new Error('api error');
      });

      req.path = '/api/error';
      req.url = new URL('http://localhost/api/error');
      await router.handle(req, socket);

      const [sockArg, statusArg, headersArg, bodyArg] = sendResponse.mock.calls[0];
      expect(sockArg).toBe(socket);
      expect(statusArg).toBe(500);
      expect(headersArg['Content-Type']).toBe('application/json');
      expect(bodyArg).toBe(JSON.stringify({ error: 'Internal Server Error' }));
      expect(sendResponse).toHaveBeenCalledTimes(1);
    });
  });

  describe('CORS and OPTIONS Handling', () => {
    test('CORS handler works directly', () => {
      handlePreflightRequest(req, socket);

      expect(socket.write).toHaveBeenCalled();
      const output = socket.write.mock.calls[0][0];

      expect(output).toContain('Access-Control-Allow-Origin: *');
      expect(output).toContain('Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS');
      expect(output).toContain(
        'Access-Control-Allow-Headers: Content-Type, Authorization, X-Requested-With, X-Filename',
      );
    });

    test('should handle OPTIONS requests with CORS headers', async () => {
      req.method = 'OPTIONS';
      req.path = '/api/test';
      req.url = new URL('http://localhost/api/test');

      await router.handle(req, socket);

      expect(sendResponse).toHaveBeenCalledWith(
        socket,
        204,
        expect.objectContaining({
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': expect.stringContaining('GET'),
          'Access-Control-Allow-Headers': expect.stringContaining('Content-Type'),
        }),
      );
    });

    test('should include matched route methods in OPTIONS response', async () => {
      // Register routes BEFORE setting up the request
      router.get('/api/test', jest.fn());
      router.post('/api/test', jest.fn());

      req.method = 'OPTIONS';
      req.path = '/api/test';
      req.url = new URL('http://localhost/api/test');

      await router.handle(req, socket);

      expect(sendResponse).toHaveBeenCalledWith(
        socket,
        204,
        expect.objectContaining({
          Allow: expect.stringContaining('GET, POST'),
        }),
      );
    });
  });

  describe('HEAD route handling', () => {
    test('should use explicit HEAD handler if present', async () => {
      // no need to spy on sendWithContext
      const sendResponseSpy = sendResponseModule.sendResponse;

      // make a HEAD request to a route with both GET and HEAD handlers
      const headReq = {
        ...req,
        method: 'HEAD',
        path: '/head-explicit',
        url: new URL('http://localhost/head-explicit'),
      };

      const headHandler = jest.fn().mockImplementation((req, sock) => {
        sendResponseModule.sendWithContext(req, sock, 204, {
          'Content-Type': 'text/plain',
          'X-Head': 'yes',
        });
      });
      const getHandler = jest.fn((req, sock) => {
        sendResponseModule.sendWithContext(
          req,
          sock,
          200,
          { 'Content-Type': 'text/plain', 'X-Head': 'yes' },
          'GET response body',
        );
      });

      router.get('/head-explicit', getHandler);
      router.head('/head-explicit', headHandler);

      await router.handle(headReq, socket);

      expect(headHandler).toHaveBeenCalled();
      expect(getHandler).not.toHaveBeenCalled();
      const [sockArg, statusArg, headersArg, bodyArg] = sendResponseSpy.mock.calls[0];
      expect(sockArg).toBe(socket);
      expect(statusArg).toBe(204);
      expect(headersArg['Content-Type']).toBe('text/plain');
      expect(headersArg['X-Head']).toBe('yes');
      expect(bodyArg).toBe(undefined); // HEAD must send empty body
    });

    test('should call GET handler for HEAD if no explicit HEAD handler', async () => {
      // no need to spyOn sendWithContext
      const sendResponseSpy = sendResponseModule.sendResponse;

      // make a HEAD request to a GET-only route
      const headReq = {
        ...req,
        method: 'HEAD',
        path: '/api/client-error-ingest',
        url: new URL('http://localhost/api/client-error-ingest'),
      };
      const getHandler = jest.fn().mockImplementation((req, sock) => {
        sendResponseModule.sendWithContext(
          req,
          sock,
          200,
          { 'Content-Type': 'text/plain', 'X-Head': 'yes' },
          'GET response body',
        );
      });
      router.get('/api/client-error-ingest', getHandler);

      await router.handle(headReq, socket);

      expect(getHandler).toHaveBeenCalled();
      expect(sendResponseSpy).toHaveBeenCalledTimes(1);

      const [sockArg, statusArg, headersArg, bodyArg] = sendResponseSpy.mock.calls[0];
      expect(sockArg).toBe(socket);
      expect(statusArg).toBe(200);
      expect(headersArg['Content-Type']).toBe('text/plain');
      expect(headersArg['X-Head']).toBe('yes');
      expect(bodyArg).toBe(undefined); // HEAD must send empty body
    });
  });
});
