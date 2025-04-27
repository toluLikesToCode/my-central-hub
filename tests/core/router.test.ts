import { router } from '../../src/core/router';
import { IncomingRequest } from '../../src/entities/http';
import { Socket } from 'net';

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

    expect(fakeSocket.write).toHaveBeenCalledWith(expect.stringContaining('404 Not Found'));
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

    expect(fakeSocket.write).toHaveBeenCalled(); // Should at least write something
  });

  it('should return 404 for wrong method', async () => {
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

    expect(fakeSocket.write).toHaveBeenCalledWith(expect.stringContaining('404 Not Found'));
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

  it('should match dynamic route /files/:filename', async () => {
    const req: IncomingRequest = {
      url: new URL('http://localhost/files/testfile.txt'),
      method: 'GET',
      path: '/files/testfile.txt',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
      query: {},
    };

    await router.handle(req, fakeSocket);

    expect(fakeSocket.write).toHaveBeenCalled(); // we should attempt to serve file
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
