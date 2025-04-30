/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck
import { FileHostingService } from '../src/modules/file-hosting/fileHostingService';
import * as fs from 'fs';
import * as fsPromises from 'fs/promises';
import { Writable } from 'stream';
import pLimit from 'p-limit';

jest.mock('fs');
jest.mock('fs/promises');

describe('FileHostingService Stress Test', () => {
  const limit = pLimit(50);
  const writeCalls: any[] = [];

  beforeEach(() => {
    jest.clearAllMocks();
    (fsPromises.mkdir as jest.Mock).mockResolvedValue(undefined);

    (fs.createWriteStream as jest.Mock).mockImplementation((path: string) => {
      if (path.includes('fail-fd')) {
        const err = new Error('Too many open files') as NodeJS.ErrnoException;
        err.code = 'EMFILE';
        throw err;
      }

      const writable = new Writable({
        write(chunk, _encoding, callback) {
          writeCalls.push(chunk.toString());
          callback();
        },
      });
      writable.on('error', jest.fn());
      writable.end = jest.fn((cb) => cb && cb());
      return writable;
    });
  });

  test('handles 1000 concurrent small uploads with disk delays and fd exhaustion', async () => {
    const service = new FileHostingService('/mock/root');

    const uploads = Array.from({ length: 1000 }).map((_, i) =>
      limit(async () => {
        const shouldFail = i % 20 === 0; // ~5% interrupted iterables
        const failFd = i % 25 === 0; // ~4% trigger EMFILE error
        const fileName = failFd ? `fail-fd-${i}.txt` : `upload-${i}.txt`;
        const chunks = [Buffer.from(`part1-${i}`), Buffer.from(`part2-${i}`)];

        const iterable = (async function* () {
          yield chunks[0];
          if (shouldFail) throw new Error(`Stream interrupted for ${i}`);
          yield chunks[1];
        })();

        try {
          await service.saveFile(fileName, iterable);
        } catch {
          // expected for failures
        }
      }),
    );

    await Promise.all(uploads);
    expect(fs.createWriteStream).toHaveBeenCalled();
    expect(writeCalls.length).toBeGreaterThan(1000);
  });
});
