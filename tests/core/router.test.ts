/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

// __tests__/router.test.ts
import { Socket } from 'net';
import { createRouter } from '../../src/core/router';
import { sendResponse } from '../../src/entities/sendResponse';
import type { IncomingRequest } from '../../src/entities/http';

jest.mock('../../src/entities/sendResponse');
jest.mock('../../src/utils/logger');

describe('Router', () => {
  let router;
  let socket: Socket;
  let req: IncomingRequest;

  beforeEach(() => {
    router = createRouter();

    socket = {
      write: jest.fn(),
      end: jest.fn(),
      destroy: jest.fn(),
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

  test('should run middleware in sequence', async () => {
    const order: string[] = [];
    router.use(async (_req, _sock, next) => {
      order.push('mw1');
      await next();
      order.push('mw1-after');
    });
    router.use(async (_req, _sock, next) => {
      order.push('mw2');
      await next();
      order.push('mw2-after');
    });

    router.any('/test', async () => {
      order.push('handler');
    });
    await router.handle(req, socket);

    expect(order).toEqual(['mw1', 'mw2', 'handler', 'mw2-after', 'mw1-after']);
  });

  test('should call matching handler for GET route', async () => {
    const handler = jest.fn();
    router.get('/test', handler);

    await router.handle(req, socket);
    expect(handler).toHaveBeenCalledWith(req, socket);
  });

  test('should respond with 404 if no route matches', async () => {
    req.path = '/unknown';
    await router.handle(req, socket);

    expect(sendResponse).toHaveBeenCalledWith(
      socket,
      404,
      { 'Content-Type': 'text/plain' },
      'Not Found',
    );
  });

  test('should respond with 405 if method does not match', async () => {
    req.method = 'POST';
    router.get('/test', jest.fn());

    await router.handle(req, socket);

    expect(sendResponse).toHaveBeenCalledWith(socket, 405, { Allow: 'GET' }, 'Method Not Allowed');
  });

  test('should extract params into req.ctx.params', async () => {
    const paramHandler = jest.fn();
    router.get('/users/:id', paramHandler);

    req.path = '/users/123';
    req.url = new URL('http://localhost/users/123');

    await router.handle(req, socket);
    expect(paramHandler).toHaveBeenCalled();
    expect(req.ctx?.params).toEqual({ id: '123' });
  });

  test('should respond with 500 on handler error', async () => {
    router.get('/error', () => {
      throw new Error('fail');
    });

    req.path = '/error';
    req.url = new URL('http://localhost/error');

    await router.handle(req, socket);

    expect(sendResponse).toHaveBeenCalledWith(
      socket,
      500,
      { 'Content-Type': 'text/plain' },
      '500 Server Error',
    );
  });

  test('should respond to OPTIONS with Allow header', async () => {
    req.method = 'OPTIONS';
    req.path = '/anything';

    await router.handle(req, socket);
    expect(sendResponse).toHaveBeenCalledWith(
      socket,
      200,
      { 'Content-Type': 'text/plain', Allow: 'GET, POST, PUT, DELETE, OPTIONS' },
      'OK',
    );
  });
});
