import { fileStreamingController } from '../../../src/modules/file-streamer';
import { IncomingRequest } from '../../../src/entities/http';
import { Socket } from 'net';
import * as fs from 'fs';
import { config } from '../../../src/config/server.config';
import { FileService } from '../../../src/modules/file-streamer/fileService';
jest.mock('fs');

describe('fileStreamingController', () => {
  const fakeSocket = {
    write: jest.fn(),
    end: jest.fn(),
  } as unknown as Socket;

  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(FileService.prototype, 'listFiles').mockResolvedValue(['video.mp4']);
    config.mediaDir = '/mocked/media'; // or whatever path you want
    (fs.existsSync as jest.Mock).mockReturnValue(true);
    (fs.readdirSync as jest.Mock).mockReturnValue(['video.mp4']);
  });

  it('should respond with 400 if no file param on stream', async () => {
    const req = {
      url: new URL('http://localhost/stream'),
      method: 'GET',
      path: '/stream',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
      query: {},
    } as IncomingRequest;
    await fileStreamingController.handleStream(req, fakeSocket);
    expect(fakeSocket.write).toHaveBeenCalledWith(
      expect.stringContaining('Missing required "file"'),
    );
  });

  it('should respond with 404 if file does not exist', async () => {
    const req = {
      url: new URL('http://localhost/stream?file=nonexistent.mp4'),
      method: 'GET',
      path: '/stream',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
      query: {},
    } as IncomingRequest;
    await fileStreamingController.handleStream(req, fakeSocket);
    expect(fakeSocket.write).toHaveBeenCalledWith(expect.stringContaining('not found'));
  });

  it('should list files correctly', async () => {
    const req: IncomingRequest = {
      url: new URL('http://localhost/files'),
      method: 'GET',
      path: '/files',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost' },
      raw: '',
      query: {},
    };

    await fileStreamingController.listFiles(req, fakeSocket);

    expect(fakeSocket.write).toHaveBeenCalledWith(expect.stringContaining('HTTP/1.1 200 OK'));
  });
});
