/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

import { fileStreamingController } from '../src/modules/file-streaming/fileStreamingController';
import { FileHostingService } from '../src/modules/file-hosting/fileHostingService';
import { sendResponse } from '../src/entities/sendResponse';
import { Readable } from 'stream';
import pLimit from 'p-limit';
import { Socket } from 'net';

jest.mock('../src/modules/file-hosting/fileHostingService');
jest.mock('../src/entities/sendResponse');

describe('File Streaming Stress Test', () => {
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

    (FileHostingService.prototype.stat as jest.Mock).mockResolvedValue({ size: 10000 });
    (FileHostingService.prototype.readFile as jest.Mock).mockResolvedValue(dummyStream());
  });

  test('200 concurrent stream requests (full + range)', async () => {
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
      } as unknown as Socket;

      tasks.push(limit(() => fileStreamingController.handleStream(req, sock)));
    }

    await Promise.all(tasks);

    expect(sendResponse).toHaveBeenCalledTimes(200);

    const codes = (sendResponse as jest.Mock).mock.calls.map(([, code]) => code);
    expect(codes.filter((c) => c === 200).length).toBeGreaterThan(80);
    expect(codes.filter((c) => c === 206).length).toBeGreaterThan(80);
  });
});
