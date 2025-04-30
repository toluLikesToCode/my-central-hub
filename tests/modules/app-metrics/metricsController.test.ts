/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

import router from '../../../src/core/router';
import { config } from '../../../src/config/server.config';
import { Socket } from 'net';
import { IncomingRequest } from '../../../src/entities/http'; // Adjust the path as needed

describe('Metrics Endpoint (gallery-generator)', () => {
  let originalMetricsFeature: boolean;

  beforeAll(() => {
    // Save and enable metrics feature for tests
    originalMetricsFeature = config.features.metrics;
    config.features.metrics = true;
  });

  afterAll(() => {
    // Restore original config
    config.features.metrics = originalMetricsFeature;
  });

  // it('should accept a valid POST and return 200', async () => {
  //   // Mock request and socket
  //   const req: IncomingRequest = {
  //     method: 'POST',
  //     path: '/api/metrics/gallery-generator',
  //     headers: { 'content-type': 'application/json' },
  //     body: Buffer.from(
  //       JSON.stringify({
  //         timestamp: new Date().toISOString(),
  //         perfNow: 123.45,
  //         memory: {
  //           usedJSHeapSize: 1024,
  //           totalJSHeapSize: 2048,
  //           jsHeapSizeLimit: 4096,
  //         },
  //         action: 'test-action',
  //         sessionId: 'session-123',
  //         batchId: 1,
  //         uploadMode: 'auto',
  //         // You can add extra fields here, as .catchall(z.unknown()) allows unknown keys
  //       }),
  //     ),
  //     ctx: {},
  //     url: new URL('http://localhost:3000/api/metrics/gallery-generator?foo=bar'), // Added fake URL
  //     query: { foo: 'bar' }, // Added fake query
  //     httpVersion: '',
  //     raw: '',
  //   };
  //   let responseCode = 0;
  //   let responseBody = '';
  //   const sock = {
  //     write: (data: string) => {
  //       const match = data.match(/HTTP\/1\.[01] (\d{3})/);
  //       if (match) responseCode = parseInt(match[1], 10);
  //       responseBody += data;
  //     },
  //     end: () => {},
  //   } as unknown as Socket;
  //   await router.handle(req, sock);
  //   expect(responseCode).toBe(200);
  //   expect(responseBody).toContain('OK');
  // });

  it('should return 404 for unknown metrics app', async () => {
    const req: Omit<IncomingRequest, 'ctx'> & { ctx: Record<string, unknown> } = {
      method: 'POST',
      path: '/api/metrics/unknown-app',
      headers: { 'content-type': 'application/json' },
      body: Buffer.from(JSON.stringify({ event: 'test' })),
      ctx: {},
      url: new URL('http://localhost/api/metrics/unknown-app'), // Added fake URL
      query: {}, // Added empty query
      httpVersion: '1.1', // Added HTTP version
      raw: '', // Added raw request string
    };
    let responseCode = 0;
    let responseBody = '';
    const sock = {
      write: (data: string) => {
        const match = data.match(/HTTP\/1\.[01] (\d{3})/);
        if (match) responseCode = parseInt(match[1], 10);
        responseBody += data;
      },
      end: () => {},
    } as unknown as Socket;
    await router.handle(req, sock);
    expect(responseCode).toBe(404);
    expect(responseBody).toContain('Not Found');
  });
});
