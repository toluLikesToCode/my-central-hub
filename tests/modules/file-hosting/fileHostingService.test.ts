/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

import { FileHostingService } from '../../../src/modules/file-hosting/fileHostingService';
import * as fs from 'fs';
import * as fsPromises from 'fs/promises';

jest.mock('fs');
jest.mock('fs/promises');

describe('FileHostingService', () => {
  let service: FileHostingService;

  beforeEach(() => {
    service = new FileHostingService('/mock/root');
  });

  test('resolveSafe throws on path traversal', () => {
    expect(() => service['resolveSafe']('../etc/passwd')).toThrow('Path traversal attempt');
  });

  test('listFiles calls readdir with safe path', async () => {
    (fsPromises.readdir as jest.Mock).mockResolvedValue(['file1.txt']);
    const result = await service.listFiles('.');
    expect(result).toEqual(['file1.txt']);
    expect(fsPromises.readdir).toHaveBeenCalledWith(expect.stringContaining('/mock/root'));
  });

  test('stat returns file stats from safe path', async () => {
    const fakeStat = { isFile: () => true };
    (fsPromises.stat as jest.Mock).mockResolvedValue(fakeStat);
    const result = await service.stat('data.txt');
    expect(result).toBe(fakeStat);
    expect(fsPromises.stat).toHaveBeenCalledWith(expect.stringContaining('/mock/root'));
  });

  test('readFile calls createReadStream with optional range', async () => {
    const createReadStream = fs.createReadStream as jest.Mock;
    await service.readFile('video.mp4', { start: 0, end: 100 });
    expect(createReadStream).toHaveBeenCalledWith(expect.stringContaining('video.mp4'), {
      start: 0,
      end: 100,
    });
  });

  test('saveFile writes data to file stream and ends it', async () => {
    const write = jest.fn();
    const end = jest.fn((cb) => cb());
    const on = jest.fn();
    const fakeStream = { write, end, on };

    (fs.createWriteStream as jest.Mock).mockReturnValue(fakeStream);
    (fsPromises.mkdir as jest.Mock).mockResolvedValue(undefined);

    const chunks = ['a', 'b'].map((x) => Buffer.from(x));
    const iterable = (async function* () {
      for (const chunk of chunks) yield chunk;
    })();

    await service.saveFile('upload.txt', iterable);

    expect(write).toHaveBeenCalledTimes(2);
    expect(end).toHaveBeenCalled();
    expect(fs.createWriteStream).toHaveBeenCalledWith(expect.stringContaining('upload.txt'));
  });
});
