/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

import { fileStreamingController } from '../../../src/modules/file-streaming/fileStreamingController';
import { sendResponse } from '../../../src/entities/sendResponse';
import { FileHostingService } from '../../../src/modules/file-hosting/fileHostingService';
import { logger } from '../../../src/utils/logger';
import { Readable } from 'stream';

jest.mock('../../../src/entities/sendResponse');
jest.mock('../../../src/modules/file-hosting/fileHostingService');
jest.mock('../../../src/utils/logger');

const createMockReadable = () => {
  const stream = new Readable();
  stream._read = () => {};
  return stream;
};

describe('fileStreamingController.handleStream', () => {
  let req: any;
  let sock: any;

  beforeEach(() => {
    jest.clearAllMocks();

    req = {
      path: '/stream',
      query: {},
      headers: {},
      url: new URL('http://localhost/stream'),
    };

    sock = {
      end: jest.fn(),
      write: jest.fn(),
      destroy: jest.fn(),
    };
  });

  test('responds 400 if no file query provided', async () => {
    await fileStreamingController.handleStream(req, sock);
    expect(sendResponse).toHaveBeenCalledWith(
      sock,
      400,
      { 'Content-Type': 'text/plain' },
      'Missing required "file" query parameter.',
    );
  });

  test('responds 404 if file not found', async () => {
    req.query.file = 'nonexistent.mp4';
    (FileHostingService.prototype.stat as jest.Mock).mockRejectedValue(new Error('not found'));

    await fileStreamingController.handleStream(req, sock);

    expect(sendResponse).toHaveBeenCalledWith(
      sock,
      404,
      { 'Content-Type': 'text/plain' },
      expect.stringContaining('not found'),
    );
  });

  test('streams full file with 200 if no range', async () => {
    req.query.file = 'video.mp4';
    (FileHostingService.prototype.stat as jest.Mock).mockResolvedValue({ size: 1000 });
    const stream = createMockReadable();
    (FileHostingService.prototype.readFile as jest.Mock).mockResolvedValue(stream);

    await fileStreamingController.handleStream(req, sock);

    expect(sendResponse).toHaveBeenCalledWith(
      sock,
      200,
      expect.objectContaining({
        'Content-Type': expect.any(String),
        'Content-Length': '1000',
      }),
      stream,
    );
  });

  test('streams partial file with 206 if valid range', async () => {
    req.query.file = 'video.mp4';
    req.headers.range = 'bytes=0-499';
    (FileHostingService.prototype.stat as jest.Mock).mockResolvedValue({ size: 1000 });
    const stream = createMockReadable();
    (FileHostingService.prototype.readFile as jest.Mock).mockResolvedValue(stream);

    await fileStreamingController.handleStream(req, sock);

    expect(sendResponse).toHaveBeenCalledWith(
      sock,
      206,
      expect.objectContaining({
        'Content-Range': 'bytes 0-499/1000',
        'Content-Length': '500',
      }),
      stream,
    );
  });

  test('responds 416 if range invalid', async () => {
    req.query.file = 'video.mp4';
    req.headers.range = 'bytes=1500-1600';
    (FileHostingService.prototype.stat as jest.Mock).mockResolvedValue({ size: 1000 });

    await fileStreamingController.handleStream(req, sock);

    expect(sendResponse).toHaveBeenCalledWith(
      sock,
      416,
      { 'Content-Type': 'text/plain' },
      '416 Range Not Satisfiable',
    );
    expect(sock.end).toHaveBeenCalled();
  });

  // test('responds 500 if stream emits error', async () => {
  //   req.query.file = 'video.mp4';
  //   (FileHostingService.prototype.stat as jest.Mock).mockResolvedValue({ size: 1000 });
  //   const stream = createMockReadable();
  //   (FileHostingService.prototype.readFile as jest.Mock).mockResolvedValue(stream);

  //   await fileStreamingController.handleStream(req, sock);
  //   stream.emit('error', new Error('stream fail'));

  //   expect(logger.error).toHaveBeenCalled();
  //   expect(sendResponse).toHaveBeenCalledWith(
  //     sock,
  //     500,
  //     { 'Content-Type': 'text/plain' },
  //     'Internal Server Error',
  //   );
  // });
});
