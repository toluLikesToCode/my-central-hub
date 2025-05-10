/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

/**
 * @deprecated This stress test file is testing a deprecated module and will be removed in the future.
 * Please use fileHosting.stress.test.ts instead for testing file operations.
 */

import { fileStreamingController } from '../src/modules/file-streaming/fileStreamingController';
import { fileHostingController } from '../src/modules/file-hosting/fileHostingController';
import { sendResponse } from '../src/entities/sendResponse';
import { Readable } from 'stream';
import pLimit from 'p-limit';
import { Socket } from 'net';

jest.mock('../src/modules/file-hosting/fileHostingService');
jest.mock('../src/modules/file-hosting/fileHostingController');
jest.mock('../src/entities/sendResponse');

describe('File Streaming Stress Test (DEPRECATED)', () => {
  const dummyStream = () => {
    const stream = new Readable();
    stream._read = () => {
      stream.push(Buffer.alloc(1024)); // 1KB chunk
      stream.push(null);
    };
    return stream;
  };

  const limit = pLimit(20); // throttle 20 at a time

  beforeEach(() => {
    jest.clearAllMocks();
    // Mock the behavior of the fileHostingController.getFile method
    // which the deprecated fileStreamingController now delegates to
    (fileHostingController.getFile as jest.Mock).mockImplementation((req, sock) => {
      sendResponse(
        sock,
        req.headers.range ? 206 : 200,
        { 'Content-Type': 'video/mp4' },
        dummyStream(),
      );
      return Promise.resolve();
    });
  });

  test('200 concurrent stream requests redirected to fileHosting', async () => {
    const tasks: Promise<void>[] = [];

    for (let i = 0; i < 200; i++) {
      const range = i % 2 === 0 ? undefined : `bytes=${i * 10}-${i * 10 + 999}`;
      const req = {
        method: 'GET',
        path: '/stream',
        query: { file: 'stress-test.mp4' },
        headers: { ...(range ? { range } : {}) } as Record<string, string>,
        httpVersion: '1.1',
        raw: '',
        url: new URL(`http://localhost/stream?file=stress-test.mp4`),
      };

      const sock = {
        write: jest.fn(),
        end: jest.fn(),
        destroy: jest.fn(),
        remoteAddress: '127.0.0.1',
      } as unknown as Socket;

      tasks.push(limit(() => fileStreamingController.handleStream(req, sock)));
    }

    await Promise.all(tasks);

    // Verify all requests were delegated to fileHostingController.getFile
    expect(fileHostingController.getFile).toHaveBeenCalledTimes(200);
  });
});
