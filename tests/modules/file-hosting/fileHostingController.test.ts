/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

import { PassThrough } from 'stream';

// Mock dependencies first, before any imports
jest.mock('../../../src/utils/logger', () => ({
  __esModule: true,
  default: {
    child: jest.fn(() => ({
      info: jest.fn(),
      warn: jest.fn(),
      error: jest.fn(),
      debug: jest.fn(),
      success: jest.fn(),
    })),
  },
}));

jest.mock('../../../src/config/server.config', () => ({
  config: {
    staticDir: '/mock/static',
    mediaDir: '/mock/media',
  },
}));

// Mock helper functions
jest.mock('../../../src/utils/helpers', () => {
  return {
    getMimeType: jest.fn((fileName) => {
      if (fileName.endsWith('.html')) return 'text/html';
      if (fileName.endsWith('.css')) return 'text/css';
      if (fileName.endsWith('.js')) return 'application/javascript';
      if (fileName.endsWith('.json')) return 'application/json';
      if (fileName.endsWith('.jpg') || fileName.endsWith('.jpeg')) return 'image/jpeg';
      if (fileName.endsWith('.png')) return 'image/png';
      if (fileName.endsWith('.svg')) return 'image/svg+xml';
      return 'text/plain';
    }),
    formatDate: jest.fn(() => 'mocked-date'),
  };
});

// Mock sendResponse
jest.mock('../../../src/entities/sendResponse', () => ({
  sendResponse: jest.fn(),
}));

// Create mock streams
const mockBrotliStream = new PassThrough();
const mockGzipStream = new PassThrough();
const mockDeflateStream = new PassThrough();
const sourceStream = new PassThrough();

// Create file hosting service mock
const mockStat = jest.fn();
const mockReadFile = jest.fn();
jest.mock('../../../src/modules/file-hosting/fileHostingService', () => {
  return {
    FileHostingService: jest.fn().mockImplementation(() => {
      return {
        stat: mockStat,
        readFile: mockReadFile,
      };
    }),
  };
});

// Mock zlib
const mockCreateBrotliCompress = jest.fn(() => mockBrotliStream);
const mockCreateGzip = jest.fn(() => mockGzipStream);
const mockCreateDeflate = jest.fn(() => mockDeflateStream);
jest.mock('zlib', () => ({
  createBrotliCompress: mockCreateBrotliCompress,
  createGzip: mockCreateGzip,
  createDeflate: mockCreateDeflate,
}));

// Now import the controller and other modules
import { sendResponse } from '../../../src/entities/sendResponse';
import { fileHostingController } from '../../../src/modules/file-hosting/fileHostingController';

describe('fileHostingController - Compression Tests', () => {
  let req;
  let sock;
  const mockFile = {
    size: 1024,
    mtime: new Date(),
    isFile: () => true,
  };

  beforeEach(() => {
    jest.clearAllMocks();

    // Reset mock streams
    mockBrotliStream.removeAllListeners();
    mockGzipStream.removeAllListeners();
    mockDeflateStream.removeAllListeners();
    sourceStream.removeAllListeners();

    // Setup mocks for each test
    mockStat.mockResolvedValue(mockFile);
    mockReadFile.mockResolvedValue(sourceStream);

    // Setup mock socket
    sock = {
      destroyed: false,
      write: jest.fn(),
      end: jest.fn(),
      destroy: jest.fn(),
      on: jest.fn(),
      remoteAddress: '127.0.0.1',
    };

    // Setup default request
    req = {
      url: 'http://example.com/api/files?file=test.html',
      path: '/api/files',
      query: { file: 'test.html' },
      headers: {
        'accept-encoding': '',
        'user-agent': 'test-agent',
      },
    };
  });

  describe('Compression Selection Logic', () => {
    test('chooses Brotli when supported and MIME type is compressible', async () => {
      req.headers['accept-encoding'] = 'gzip, deflate, br';

      await fileHostingController.getFile(req, sock);

      expect(mockCreateBrotliCompress).toHaveBeenCalled();
      expect(mockCreateGzip).not.toHaveBeenCalled();
      expect(mockCreateDeflate).not.toHaveBeenCalled();

      expect(sendResponse).toHaveBeenCalledWith(
        sock,
        200,
        expect.objectContaining({
          'Content-Encoding': 'br',
          Vary: 'Accept-Encoding',
        }),
        expect.any(Object),
      );
    });

    test('chooses Gzip when Brotli not supported but Gzip is', async () => {
      req.headers['accept-encoding'] = 'gzip, deflate';

      await fileHostingController.getFile(req, sock);

      expect(mockCreateBrotliCompress).not.toHaveBeenCalled();
      expect(mockCreateGzip).toHaveBeenCalled();
      expect(mockCreateDeflate).not.toHaveBeenCalled();

      expect(sendResponse).toHaveBeenCalledWith(
        sock,
        200,
        expect.objectContaining({
          'Content-Encoding': 'gzip',
          Vary: 'Accept-Encoding',
        }),
        expect.any(Object),
      );
    });

    test('chooses Deflate when only Deflate is supported', async () => {
      req.headers['accept-encoding'] = 'deflate';

      await fileHostingController.getFile(req, sock);

      expect(mockCreateBrotliCompress).not.toHaveBeenCalled();
      expect(mockCreateGzip).not.toHaveBeenCalled();
      expect(mockCreateDeflate).toHaveBeenCalled();

      expect(sendResponse).toHaveBeenCalledWith(
        sock,
        200,
        expect.objectContaining({
          'Content-Encoding': 'deflate',
          Vary: 'Accept-Encoding',
        }),
        expect.any(Object),
      );
    });

    test('uses no compression when no algorithm is supported', async () => {
      req.headers['accept-encoding'] = '';

      await fileHostingController.getFile(req, sock);

      expect(mockCreateBrotliCompress).not.toHaveBeenCalled();
      expect(mockCreateGzip).not.toHaveBeenCalled();
      expect(mockCreateDeflate).not.toHaveBeenCalled();

      expect(sendResponse).toHaveBeenCalledWith(
        sock,
        200,
        expect.not.objectContaining({
          'Content-Encoding': expect.anything(),
          Vary: 'Accept-Encoding',
        }),
        expect.any(Object),
      );

      expect(sendResponse).toHaveBeenCalledWith(
        sock,
        200,
        expect.objectContaining({
          'Content-Length': String(mockFile.size),
        }),
        expect.any(Object),
      );
    });

    test('uses no compression for non-compressible MIME types', async () => {
      req.headers['accept-encoding'] = 'gzip, deflate, br';
      req.query.file = 'image.jpg'; // Image file - not compressible

      await fileHostingController.getFile(req, sock);

      expect(mockCreateBrotliCompress).not.toHaveBeenCalled();
      expect(mockCreateGzip).not.toHaveBeenCalled();
      expect(mockCreateDeflate).not.toHaveBeenCalled();

      expect(sendResponse).toHaveBeenCalledWith(
        sock,
        200,
        expect.not.objectContaining({
          'Content-Encoding': expect.anything(),
        }),
        expect.any(Object),
      );
    });
  });

  describe('Range Requests and Compression', () => {
    test('does not use compression for range requests', async () => {
      req.headers['accept-encoding'] = 'gzip, deflate, br';
      req.headers['range'] = 'bytes=0-100';

      // For range requests, we need a different mock implementation
      const rangeStream = new PassThrough();
      mockReadFile.mockImplementation((path, range) => {
        if (range) {
          return Promise.resolve(rangeStream);
        }
        return Promise.resolve(sourceStream);
      });

      await fileHostingController.getFile(req, sock);

      expect(mockCreateBrotliCompress).not.toHaveBeenCalled();
      expect(mockCreateGzip).not.toHaveBeenCalled();
      expect(mockCreateDeflate).not.toHaveBeenCalled();

      expect(sendResponse).toHaveBeenCalledWith(
        expect.anything(),
        206,
        expect.objectContaining({
          'Content-Range': expect.stringContaining('bytes'),
        }),
        expect.any(Object),
      );

      expect(sendResponse).toHaveBeenCalledWith(
        expect.anything(),
        206,
        expect.not.objectContaining({
          'Content-Encoding': expect.anything(),
          Vary: 'Accept-Encoding',
        }),
        expect.any(Object),
      );
    });
  });

  describe('Streaming Functionality', () => {
    test('correctly pipes source through compression stream', async () => {
      req.headers['accept-encoding'] = 'br';

      // Create a spy on the pipe method
      const pipeSpy = jest.spyOn(sourceStream, 'pipe');

      await fileHostingController.getFile(req, sock);

      expect(pipeSpy).toHaveBeenCalledWith(mockBrotliStream);

      expect(sendResponse).toHaveBeenCalledWith(
        expect.anything(),
        200,
        expect.anything(),
        mockBrotliStream,
      );
    });
  });
});
