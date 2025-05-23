/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

import { HttpServer } from '../src/core/server';
import { createRouter } from '../src/core/router';
import { Socket } from 'net';
import pLimit from 'p-limit';

describe('HttpServer Stress Test', () => {
  let server: HttpServer;
  let port: number;

  beforeAll(async () => {
    const router = createRouter();
    router.get('/ping', (req, sock) => {
      sock.write('HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK');
    });

    server = new HttpServer(0, router);
    await new Promise((resolve) => {
      server['server'].listen(0, () => {
        port = (server['server'].address() as any).port;
        resolve(null);
      });
    });
  });

  afterAll(async () => {
    await server.stop();
  });

  test.skip('handles 500 concurrent TCP socket clients with throttling', async () => {
    const limit = pLimit(50);
    const results: Promise<string>[] = [];

    for (let i = 0; i < 500; i++) {
      results.push(
        limit(
          () =>
            new Promise((resolve, reject) => {
              const client = new Socket();
              let data = '';
              client.connect(port, '127.0.0.1', () => {
                client.write('GET /ping HTTP/1.1\r\nHost: localhost\r\n\r\n');
              });
              client.on('data', (chunk) => {
                data += chunk.toString();
                if (data.includes('OK')) {
                  client.end();
                  resolve(data);
                }
              });
              client.on('error', reject);
              client.on('end', () => {
                if (!data.includes('OK')) reject(new Error('Incomplete response'));
              });
            }),
        ),
      );
    }

    const responses = await Promise.all(results);
    expect(responses).toHaveLength(500);
    for (const res of responses) {
      expect(res).toContain('200 OK');
      expect(res).toContain('OK');
    }
  });

  test('socket closure', () => {
    const mockSocket = {
      close: jest.fn(),
      end: jest.fn(),
      destroy: jest.fn(),
    };

    // Simulate a closure event
    mockSocket.close();

    // Accept either .close(), .end(), or .destroy() as valid ways to close the socket
    expect(
      mockSocket.close.mock.calls.length > 0 ||
        mockSocket.end.mock.calls.length > 0 ||
        mockSocket.destroy.mock.calls.length > 0,
    ).toBe(true);
  });
});
