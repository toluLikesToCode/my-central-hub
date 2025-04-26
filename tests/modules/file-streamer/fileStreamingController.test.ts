import { fileStreamingController } from '../../../src/modules/file-streamer';
import { IncomingRequest } from '../../../src/entities/http';
import { Socket } from 'net';

describe('fileStreamingController', () => {
  const fakeSocket = {
    write: jest.fn(),
    end: jest.fn(),
  } as unknown as Socket;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should respond with 400 if no file param on stream', async () => {
    const req: IncomingRequest = {
      method: 'GET',
      path: '/stream',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
    };

    await fileStreamingController.handleStream(req, fakeSocket);

    expect(fakeSocket.write).toHaveBeenCalledWith(expect.stringContaining('400 Bad Request'));
  });

  it('should list files correctly', async () => {
    const req: IncomingRequest = {
      method: 'GET',
      path: '/files',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
    };

    await fileStreamingController.listFiles(req, fakeSocket);

    expect(fakeSocket.write).toHaveBeenCalledWith(expect.stringContaining('HTTP/1.1 200 OK'));
  });
});
