/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

import * as fs from 'fs';
import * as fsPromises from 'fs/promises';
import { Readable } from 'stream';

jest.mock('fs');
jest.mock('fs/promises');
jest.mock('../../../src/config/server.config', () => ({
  config: {
    fileCache: {
      enabled: true,
      maxSize: 50 * 1024 * 1024,
      maxAge: 60000,
    },
    mediaDir: '/mock/root',
  },
}));

// Note: Mock the logger BEFORE importing the service that uses it
const mockLoggerInstance = {
  info: jest.fn(),
  error: jest.fn(),
  warn: jest.fn(),
  debug: jest.fn(),
  child: jest.fn().mockReturnThis(), // Preserve chained calls
  success: jest.fn(),
};
jest.mock('../../../src/utils/logger', () => ({
  __esModule: true, // Support ES module default export
  default: mockLoggerInstance,
}));

// Now import the service with logger mock in place
import { FileHostingService } from '../../../src/modules/file-hosting/fileHostingService';

describe('FileHostingService', () => {
  let service: FileHostingService;
  const mockDateNow = Date.now();

  beforeEach(() => {
    jest.clearAllMocks();
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
    const fakeStat = {
      isFile: () => true,
      size: 1234,
      mtime: new Date(mockDateNow),
      mtimeMs: mockDateNow,
    };
    (fsPromises.stat as jest.Mock).mockResolvedValue(fakeStat);
    const result = await service.stat('data.txt');
    expect(result).toBe(fakeStat);
    expect(fsPromises.stat).toHaveBeenCalledWith(expect.stringContaining('/mock/root/data.txt'));
  });

  test('readFile streams directly for ranges, bypassing cache', async () => {
    const mockStream = new Readable();
    (fs.createReadStream as jest.Mock).mockReturnValue(mockStream);

    const stream = await service.readFile('video.mp4', { start: 0, end: 100 });
    expect(fs.createReadStream).toHaveBeenCalledWith(expect.stringContaining('video.mp4'), {
      start: 0,
      end: 100,
    });
    expect(stream).toBe(mockStream);
    expect(fsPromises.readFile).not.toHaveBeenCalled();
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
    expect(fs.createWriteStream).toHaveBeenCalledWith(
      expect.stringContaining('/mock/root/upload.txt'),
    );
  });

  test('readFile caches small files on first read (cache miss)', async () => {
    const fileContent = Buffer.from('small file content');
    const fileSize = fileContent.length;
    const filePath = 'small.txt';
    const fakeStat = { size: fileSize, mtime: new Date(mockDateNow), mtimeMs: mockDateNow };

    (fsPromises.stat as jest.Mock).mockResolvedValue(fakeStat);
    (fsPromises.readFile as jest.Mock).mockResolvedValue(fileContent);

    const cacheSetSpy = jest.spyOn(service['fileCache'], 'set');

    const stream = await service.readFile(filePath);
    const receivedData = await streamToString(stream);

    expect(fsPromises.stat).toHaveBeenCalledWith(expect.stringContaining(filePath));
    expect(fsPromises.readFile).toHaveBeenCalledWith(expect.stringContaining(filePath));
    expect(cacheSetSpy).toHaveBeenCalledWith(filePath, {
      buffer: fileContent,
      mtime: mockDateNow,
      size: fileSize,
      mimeType: 'text/plain',
    });
    expect(receivedData).toBe(fileContent.toString());

    const stats = service.getCacheStats();
    expect(stats.misses).toBe(1);
    expect(stats.hits).toBe(0);
    expect(stats.size).toBe(fileSize);
  });

  test('readFile returns cached file if mtime matches (cache hit)', async () => {
    const fileContent = Buffer.from('cached content');
    const fileSize = fileContent.length;
    const filePath = 'cached.txt';
    const fakeStat = { size: fileSize, mtime: new Date(mockDateNow), mtimeMs: mockDateNow };

    service['fileCache'].set(filePath, {
      buffer: fileContent,
      mtime: mockDateNow,
      size: fileSize,
      mimeType: 'text/plain',
    });

    (fsPromises.stat as jest.Mock).mockResolvedValue(fakeStat);
    const cacheGetSpy = jest.spyOn(service['fileCache'], 'get');

    const stream = await service.readFile(filePath);
    const receivedData = await streamToString(stream);

    expect(fsPromises.stat).toHaveBeenCalledWith(expect.stringContaining(filePath));
    expect(cacheGetSpy).toHaveBeenCalledWith(filePath);
    expect(fsPromises.readFile).not.toHaveBeenCalled();
    expect(fs.createReadStream).not.toHaveBeenCalled();
    expect(receivedData).toBe(fileContent.toString());

    const stats = service.getCacheStats();
    expect(stats.hits).toBe(1);
    expect(stats.misses).toBe(0);
  });

  test('readFile reads from disk if cached file mtime differs (cache miss)', async () => {
    const oldContent = Buffer.from('old cached content');
    const newContent = Buffer.from('new disk content');
    const filePath = 'updated.txt';
    const oldTime = mockDateNow - 10000;
    const newTime = mockDateNow;
    const fakeStat = { size: newContent.length, mtime: new Date(newTime), mtimeMs: newTime };

    service['fileCache'].set(filePath, {
      buffer: oldContent,
      mtime: oldTime,
      size: oldContent.length,
      mimeType: 'text/plain',
    });

    (fsPromises.stat as jest.Mock).mockResolvedValue(fakeStat);
    (fsPromises.readFile as jest.Mock).mockResolvedValue(newContent);
    const cacheSetSpy = jest.spyOn(service['fileCache'], 'set');

    const stream = await service.readFile(filePath);
    const receivedData = await streamToString(stream);

    expect(fsPromises.stat).toHaveBeenCalled();
    expect(fsPromises.readFile).toHaveBeenCalled();
    expect(cacheSetSpy).toHaveBeenCalledWith(
      filePath,
      expect.objectContaining({ mtime: newTime, buffer: newContent }),
    );
    expect(receivedData).toBe(newContent.toString());

    const stats = service.getCacheStats();
    expect(stats.misses).toBe(1);
  });

  test('readFile streams directly from disk for large files, bypassing cache', async () => {
    const largeFileSize = 60 * 1024 * 1024;
    const filePath = 'large_video.mp4';
    const fakeStat = { size: largeFileSize, mtime: new Date(mockDateNow), mtimeMs: mockDateNow };
    const mockDiskStream = new Readable();

    (fsPromises.stat as jest.Mock).mockResolvedValue(fakeStat);
    (fs.createReadStream as jest.Mock).mockReturnValue(mockDiskStream);
    const cacheSetSpy = jest.spyOn(service['fileCache'], 'set');

    const stream = await service.readFile(filePath);

    expect(fsPromises.stat).toHaveBeenCalled();
    expect(fs.createReadStream).toHaveBeenCalledWith(expect.stringContaining(filePath));
    expect(fsPromises.readFile).not.toHaveBeenCalled();
    expect(cacheSetSpy).not.toHaveBeenCalled();
    expect(stream).toBe(mockDiskStream);

    const stats = service.getCacheStats();
    expect(stats.misses).toBe(1);
  });

  test('saveFile invalidates cache for the saved file', async () => {
    const filePath = 'to_overwrite.txt';
    service['fileCache'].set(filePath, {
      buffer: Buffer.from('old'),
      mtime: mockDateNow,
      size: 3,
      mimeType: 'text/plain',
    });
    const cacheDeleteSpy = jest.spyOn(service['fileCache'], 'delete');

    const fakeStream = { write: jest.fn(), end: jest.fn((cb) => cb()), on: jest.fn() };
    (fs.createWriteStream as jest.Mock).mockReturnValue(fakeStream);
    (fsPromises.mkdir as jest.Mock).mockResolvedValue(undefined);

    const iterable = (async function* () {
      yield Buffer.from('new');
    })();
    await service.saveFile(filePath, iterable);

    expect(cacheDeleteSpy).toHaveBeenCalledWith(filePath);
  });

  test('deleteFile removes file and invalidates cache', async () => {
    const filePath = 'to_delete.txt';
    service['fileCache'].set(filePath, {
      buffer: Buffer.from('delete me'),
      mtime: mockDateNow,
      size: 9,
      mimeType: 'text/plain',
    });
    const cacheDeleteSpy = jest.spyOn(service['fileCache'], 'delete');
    (fsPromises.unlink as jest.Mock).mockResolvedValue(undefined);

    await service.deleteFile(filePath);

    expect(fsPromises.unlink).toHaveBeenCalledWith(expect.stringContaining(filePath));
    expect(cacheDeleteSpy).toHaveBeenCalledWith(filePath);
  });
});

// Helper function to convert a Readable stream to a string
async function streamToString(stream: Readable): Promise<string> {
  const chunks: Buffer[] = [];
  return new Promise((resolve, reject) => {
    stream.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
    stream.on('error', (err) => reject(err));
    stream.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
  });
}
