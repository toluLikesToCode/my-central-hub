let fileStreamingController: any;
let fs: any;
let fsPromises: any;
let config: any;
import { IncomingRequest } from '../../../src/entities/http';
import { Socket } from 'net';
import { Readable } from 'stream';
jest.mock('fs');
jest.mock('fs/promises');

const fakeStream = new Readable({ read() {} }); // a real Readable stream

// Dynamically load controller after resetting modules and config
describe('fileStreamingController', () => {
  const fakeSocket = {
    write: jest.fn(),
    end: jest.fn(),
  } as unknown as Socket;

  beforeEach(() => {
    jest.resetModules();
    jest.clearAllMocks();
    // require fresh modules after reset
    fs = require('fs');
    fsPromises = require('fs/promises');
    config = require('../../../src/config/server.config').config;
    config.mediaDir = '/mocked/media';
    // mock fs methods
    fs.existsSync.mockReturnValue(true);
    fsPromises.readdir.mockResolvedValue(['video.mp4']);
    fsPromises.stat.mockResolvedValue({ size: 1000, isFile: () => true });
    fs.createReadStream.mockReturnValue(fakeStream);
    // load controller after mocks and config override
    fileStreamingController =
      require('../../../src/modules/file-streamer/fileStreamingController').fileStreamingController;
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
    fs.existsSync.mockReturnValue(false);
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

  it('should stream file if it exists', async () => {
    const req = {
      url: new URL('http://localhost/stream?file=video.mp4'),
      method: 'GET',
      path: '/stream',
      httpVersion: 'HTTP/1.1',
      headers: { host: 'localhost', range: 'bytes=0-499' },
      raw: '',
      query: { file: 'video.mp4' },
    } as IncomingRequest;

    await fileStreamingController.handleStream(req, fakeSocket);

    expect(fakeSocket.write).toHaveBeenCalledWith(
      expect.stringContaining('HTTP/1.1 206 Partial Content'),
    );
    expect(fs.createReadStream).toHaveBeenCalledWith(expect.stringContaining('video.mp4'), {
      start: 0,
      end: 499,
    });
  });
});
