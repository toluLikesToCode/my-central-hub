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

afterAll(async () => {
  await httpServer.stop();
});
