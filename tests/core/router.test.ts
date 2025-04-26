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
      method: 'GET',
      path: '/doesnotexist',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
    };

    await router.handle(req, fakeSocket);

    expect(fakeSocket.write).toHaveBeenCalledWith(expect.stringContaining('404 Not Found'));
  });

  it('should handle a valid /files GET request', async () => {
    const req: IncomingRequest = {
      method: 'GET',
      path: '/files',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
    };

    await router.handle(req, fakeSocket);

    expect(fakeSocket.write).toHaveBeenCalled(); // Should at least write something
  });
});
