/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck
import { PassThrough } from 'stream';

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

// Mock sendResponse while keeping sendWithContext real
const mockSendResponse = jest.fn();
jest.mock('../../../src/entities/sendResponse', () => {
  const actual = jest.requireActual('../../../src/entities/sendResponse');
  return {
    ...actual,
    // Mock sendResponse but keep sendWithContext real
    sendResponse: mockSendResponse,
    // Override sendWithContext to use our mock
    sendWithContext: jest.fn((req, sock, status, headers, body) => {
      console.log('sendWithContext called with status:', status);
      console.log('Headers:', headers);
      // Forward to our mock sendResponse
      mockSendResponse(sock, status, headers, body);
    }),
  };
});

// Create mock streams
const mockBrotliStream = new PassThrough();
const mockGzipStream = new PassThrough();
const mockDeflateStream = new PassThrough();
const sourceStream = new PassThrough();

// Create file hosting service mock
const mockStat = jest.fn();
const mockReadFile = jest.fn();
const mockFindFileByName = jest.fn();
jest.mock('../../../src/modules/file-hosting/fileHostingService', () => {
  return {
    FileHostingService: jest.fn().mockImplementation(() => {
      return {
        stat: mockStat,
        readFile: mockReadFile,
        findFileByName: mockFindFileByName,
      };
    }),
  };
});

// Mock FileHostingStatsHelper
const mockInitialize = jest.fn().mockResolvedValue(undefined);
const mockGetStatsByPath = jest.fn().mockResolvedValue(null);
const mockDeleteFileStats = jest.fn().mockResolvedValue(true);
const mockSaveFileStats = jest.fn().mockResolvedValue(1);
jest.mock('../../../src/modules/file-hosting/fileHostingStatsHelper', () => {
  return {
    FileHostingStatsHelper: jest.fn().mockImplementation(() => {
      return {
        initialize: mockInitialize,
        getStatsByPath: mockGetStatsByPath,
        deleteFileStats: mockDeleteFileStats,
        saveFileStats: mockSaveFileStats,
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
// Mock the controller initialization promise
jest.mock('../../../src/modules/file-hosting/fileHostingController', () => {
  const actual = jest.requireActual('../../../src/modules/file-hosting/fileHostingController');
  return {
    ...actual,
    // Mock the initialization promise to be already resolved
    __fileHostingStatsHelperInit: Promise.resolve(),
  };
});
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

    // Setup default request - IMPORTANT: We add a ctx with our mock sendResponse
    req = {
      url: 'http://example.com/api/files?file=test.html',
      path: '/api/files',
      query: { file: 'test.html' },
      headers: {
        'accept-encoding': '',
        'user-agent': 'test-agent',
      },
      // Add ctx with sendResponse to ensure our mock is used
      ctx: {
        sendResponse: mockSendResponse,
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

      expect(mockSendResponse).toHaveBeenCalledWith(
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

      expect(mockSendResponse).toHaveBeenCalledWith(
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

      expect(mockSendResponse).toHaveBeenCalledWith(
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

      expect(mockSendResponse).toHaveBeenCalledWith(
        sock,
        200,
        expect.not.objectContaining({
          'Content-Encoding': expect.anything(),
          Vary: 'Accept-Encoding',
        }),
        expect.any(Object),
      );

      expect(mockSendResponse).toHaveBeenCalledWith(
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

      expect(mockSendResponse).toHaveBeenCalledWith(
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

      expect(mockSendResponse).toHaveBeenCalledWith(
        expect.anything(),
        206,
        expect.objectContaining({
          'Content-Range': expect.stringContaining('bytes'),
        }),
        expect.any(Object),
      );

      expect(mockSendResponse).toHaveBeenCalledWith(
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

      expect(mockSendResponse).toHaveBeenCalledWith(
        expect.anything(),
        200,
        expect.anything(),
        mockBrotliStream,
      );
    });
  });

  describe('resolveFileName', () => {
    test('normalizes paths correctly by removing parent segments', async () => {
      // Mock request with a path containing parent traversal segments
      const reqNorm = {
        ctx: { params: { filename: 'folder/../test.html' } },
        url: '/api/files/folder/../test.html',
        query: {},
        headers: {},
      };
      const sockNorm = {
        destroyed: false,
        write: jest.fn(),
        end: jest.fn(),
        destroy: jest.fn(),
        on: jest.fn(),
        remoteAddress: '127.0.0.1',
      };

      // Ensure stat succeeds on normalized name
      mockStat.mockResolvedValue({ size: 0, mtime: new Date(), isFile: () => true });

      const result = await fileHostingController.resolveFileName(reqNorm, sockNorm);
      expect(result).toBe('test.html');
      // Should not send an error response
      expect(mockSendResponse).not.toHaveBeenCalled();
    });

    test('rejects paths with parent traversal segments and sends 400', async () => {
      const reqReject = {
        ctx: { params: { filename: '../../secret.txt' } },
        url: '/api/files/../../secret.txt',
        query: {},
        headers: {},
      };
      const sockReject = {
        destroyed: false,
        write: jest.fn(),
        end: jest.fn(),
        destroy: jest.fn(),
        on: jest.fn(),
        remoteAddress: '127.0.0.1',
      };

      mockSendResponse.mockClear();
      const resultReject = await fileHostingController.resolveFileName(reqReject, sockReject);
      expect(resultReject).toBeUndefined();
      // Should have sent a 400 plain response for invalid path
      expect(mockSendResponse).toHaveBeenCalledWith(
        sockReject,
        400,
        expect.objectContaining({ 'Content-Type': 'text/plain' }),
        'Invalid file path.',
      );
    });
  });
});
