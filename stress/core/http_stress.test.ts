import { createConnection } from 'net';
import { config } from '../../src/config/server.config'; // assuming your port config
import { HttpServer } from '../../src/core/server';

jest.setTimeout(20000); // Increase timeout for all tests in this file
config.bodyTimeoutMs = 10000; // Set body timeout to 10 seconds
config.headerTimeoutMs = 10000; // Set header timeout to 10 seconds
let server: HttpServer;

beforeAll(async () => {
  server = new HttpServer(config.port);
  server.start();
  // Wait briefly to ensure server is ready
  await new Promise((res) => setTimeout(res, 300));
});

afterEach(() => {
  jest.clearAllTimers();
});

afterAll(async () => {
  await server.stop();
  // Ensure all sockets are closed using the new public method
  server.destroySockets();
});

function sendRawRequest(payload: string | Buffer, expectClose: boolean = true): Promise<string> {
  return new Promise((resolve, reject) => {
    const client = createConnection({ port: config.port }, () => {
      client.write(payload);
    });

    let data = '';
    let timeout: NodeJS.Timeout | null = null;

    client.on('data', (chunk) => {
      data += chunk.toString();
    });

    client.on('end', () => {
      if (timeout) clearTimeout(timeout);
      resolve(data);
    });

    client.on('error', (err) => {
      if (timeout) clearTimeout(timeout);
      reject(err);
    });

    if (!expectClose) {
      timeout = setTimeout(() => client.end(), 3000); // manual timeout fallback
    }
  });
}

describe('🔥 HTTP Server TCP Stress Tests', () => {
  it('handles simple GET properly', async () => {
    const response = await sendRawRequest('GET /files HTTP/1.1\r\nHost: localhost\r\n\r\n');
    expect(response).toMatch(/HTTP\/1\.1 200 OK/);
  });

  it('handles fragmented headers across TCP packets', async () => {
    const sock = createConnection({ port: config.port });

    sock.write('GET /files HT');
    await new Promise((r) => setTimeout(r, 50));
    sock.write('TP/1.1\r\nHo');
    await new Promise((r) => setTimeout(r, 50));
    sock.write('st: localhost\r\n\r\n');

    let response = '';
    sock.on('data', (chunk) => {
      response += chunk.toString();
    });

    const res = await new Promise<string>((resolve) => {
      sock.on('end', () => resolve(response));
    });

    expect(res).toMatch(/HTTP\/1\.1 200 OK/);
  });

  it('handles invalid request line (bad client)', async () => {
    const response = await sendRawRequest('BADREQUEST\r\nHost: localhost\r\n\r\n');
    expect(response).toMatch(/HTTP\/1.1 400 Bad Request/);
  });

  it('rejects massive headers flood', async () => {
    const massiveHeaders = Array(1200).fill('X-Flood: yes').join('\r\n');
    const request = `GET / HTTP/1.1\r\n${massiveHeaders}\r\n\r\n`;
    const response = await sendRawRequest(request);
    expect(response).toMatch(/HTTP\/1.1 400 Bad Request/);
  });

  it('rejects huge body POST (over limit)', async () => {
    const bigBody = 'A'.repeat(12 * 1024 * 1024); // 12MB
    const request = `POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: ${bigBody.length}\r\n\r\n${bigBody}`;
    const response = await sendRawRequest(request);
    expect(response).toMatch(/HTTP\/1.1 400 Bad Request/);
  });

  it('handles chunked upload streamed slowly', async () => {
    const sock = createConnection({ port: config.port });
    const parts = [
      'POST /upload HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n',
      '5\r\nHello\r\n',
      '6\r\n World\r\n',
      '0\r\n\r\n',
    ];

    for (const p of parts) {
      sock.write(p);
      await new Promise((r) => setTimeout(r, 100));
    }

    let response = '';
    sock.on('data', (chunk) => {
      response += chunk.toString();
    });

    const res = await new Promise<string>((resolve) => {
      sock.on('end', () => resolve(response));
    });

    expect(res).toMatch(/HTTP\/1\.1/);
    expect(res).not.toMatch(/400/);
  });

  // Add a test for header folding
  it('handles folded headers correctly', async () => {
    const foldedHeaderRequest = `GET / HTTP/1.1\r\nHost: localhost\r\nX-Folded: part1\r\n\tpart2\r\n\r\n`;
    const response = await sendRawRequest(foldedHeaderRequest);
    expect(response).toMatch(/HTTP\/1\.1 404 Not Found/);
  });

  it('supports HTTP/1.1 pipelined requests', async () => {
    const payload =
      'GET /files HTTP/1.1\r\nHost: localhost\r\n\r\n' +
      'GET /files HTTP/1.1\r\nHost: localhost\r\n\r\n';
    const response = await sendRawRequest(payload, false);
    const occurrences = (response.match(/HTTP\/1\.1/gi) || []).length;
    expect(occurrences).toBe(2);
  });
});
