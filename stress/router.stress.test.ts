/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck
import { createRouter } from '../src/core/router';
import { Socket } from 'net';

describe('Router Stress Test', () => {
  let router: ReturnType<typeof createRouter>;
  let mockSocket: Socket;

  beforeEach(() => {
    router = createRouter();
    mockSocket = {
      write: jest.fn(),
      end: jest.fn(),
      destroy: jest.fn(),
    } as unknown as Socket;
  });

  test('handles 100 rapid requests across different routes', async () => {
    const results: string[] = [];
    const handler = jest.fn((req) => {
      results.push(`${req.method} ${req.path}`);
    });

    const middleware = jest.fn(async (req, sock, next) => {
      await next();
    });

    router.use(middleware);
    router.get('/a', handler);
    router.get('/b', handler);
    router.get('/c/:id', handler);

    const requests = Array.from({ length: 100 }).map((_, i) => {
      const path = i % 3 === 0 ? '/a' : i % 3 === 1 ? '/b' : `/c/${i}`;
      return {
        method: 'GET',
        path,
        query: {},
        headers: {},
        url: new URL(`http://localhost${path}`),
        ctx: {},
        httpVersion: 'HTTP/1.1',
        headersMap: new Map(),
        raw: '',
        invalid: false,
      };
    });

    await Promise.all(requests.map((req) => router.handle(req, mockSocket)));

    expect(handler).toHaveBeenCalledTimes(100);
    expect(middleware).toHaveBeenCalledTimes(100);
  });
});
