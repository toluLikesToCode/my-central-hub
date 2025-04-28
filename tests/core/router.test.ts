import { router } from '../../src/core/router';
import { IncomingRequest } from '../../src/entities/http';
import { Socket } from 'net';

// Mock the file-streaming controller so router tests stay focused
jest.mock('../../src/modules/file-streamer/fileStreamingController', () => ({
  fileStreamingController: {
    listFiles: jest.fn(async (_req, sock) => {
      sock.write('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n');
    }),
    handleStream: jest.fn(async (_req, sock) => {
      sock.write('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n');
    }),
  },
}));

describe('Router', () => {
  const fakeSocket = {
    write: jest.fn(),
    end: jest.fn(),
  } as unknown as Socket;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should return 404 for unknown route', async () => {
    const req: IncomingRequest = {
      url: new URL('http://localhost/doesnotexist'),
      method: 'GET',
      path: '/doesnotexist',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
      query: {},
    };

    await router.handle(req, fakeSocket);

    expect(fakeSocket.write).toHaveBeenCalledWith(
      expect.stringContaining('HTTP/1.1 404 Not Found'),
    );
  });

  it('should handle a valid /files GET request', async () => {
    const req: IncomingRequest = {
      url: new URL('http://localhost/files'),
      method: 'GET',
      path: '/files',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
      query: {},
    };

    await router.handle(req, fakeSocket);

    expect(fakeSocket.write).toHaveBeenCalledWith(expect.stringContaining('HTTP/1.1 200 OK'));
  });

  it('should return 405 Method Not Allowed for wrong method', async () => {
    const req: IncomingRequest = {
      url: new URL('http://localhost/files'),
      method: 'POST', // Should be GET
      path: '/files',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
      query: {},
    };

    await router.handle(req, fakeSocket);

    expect(fakeSocket.write).toHaveBeenCalledWith(
      expect.stringContaining('HTTP/1.1 405 Method Not Allowed'),
    );
  });

  it('should handle missing path safely', async () => {
    const req: IncomingRequest = {
      url: new URL('http://localhost/'),
      method: 'GET',
      path: undefined as unknown as string, // force a bad input
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
      query: {},
    };

    await router.handle(req, fakeSocket);

    expect(fakeSocket.write).toHaveBeenCalledWith(expect.stringContaining('400 Bad Request'));
  });

  it('should handle /stream route with file query parameter', async () => {
    const req: IncomingRequest = {
      url: new URL('http://localhost/stream?file=testfile.txt'),
      method: 'GET',
      path: '/stream',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
      query: { file: 'testfile.txt' },
    };

    await router.handle(req, fakeSocket);

    expect(fakeSocket.write).toHaveBeenCalled(); // should attempt to stream the requested file
  });

  it('should return 500 if handler throws', async () => {
    router.get('/error', async () => {
      throw new Error('fail');
    });
    const validReq: IncomingRequest = {
      url: new URL('http://localhost/error'),
      method: 'GET',
      path: '/error',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
      query: {},
    };
    const req = { ...validReq, path: '/error' };

    await router.handle(req, fakeSocket);

    expect(fakeSocket.write).toHaveBeenCalledWith(expect.stringContaining('500 Server Error'));
  });
});
