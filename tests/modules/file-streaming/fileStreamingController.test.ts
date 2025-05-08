/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

/**
 * @deprecated This test file tests a deprecated module and will be removed in the future.
 * The file-streaming module has been replaced by the file-hosting module.
 * Future tests should use the file-hosting controller tests instead.
 */

jest.mock('../../../src/entities/sendResponse');
jest.mock('../../../src/modules/file-hosting/fileHostingService');
jest.mock('../../../src/utils/logger');
jest.mock('../../../src/modules/file-hosting/fileHostingController');

// Mock the deprecation logger
jest.mock('../../../src/modules/file-streaming/fileStreamingController', () => {
  const originalModule = jest.requireActual(
    '../../../src/modules/file-streaming/fileStreamingController',
  );

  // Create a fake deprecation logger with all needed methods
  const mockDeprecationLogger = {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  };

  // Override the exported object with our instrumented version
  return {
    fileStreamingController: {
      ...originalModule.fileStreamingController,
      // Inject our mock logger for tests
      deprecationLogger: mockDeprecationLogger,
    },
  };
});

import { fileStreamingController } from '../../../src/modules/file-streaming/fileStreamingController';
import { fileHostingController } from '../../../src/modules/file-hosting/fileHostingController';
import { sendResponse } from '../../../src/entities/sendResponse';
import { FileHostingService } from '../../../src/modules/file-hosting/fileHostingService';
import logger from '../../../src/utils/logger';
import { Readable } from 'stream';

const createMockReadable = () => {
  const stream = new Readable();
  stream._read = () => {};
  return stream;
};

describe('fileStreamingController.handleStream (DEPRECATED)', () => {
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

  test('logs deprecation warning', async () => {
    req.query.file = 'test.mp4';
    await fileStreamingController.handleStream(req, sock);
    expect(fileStreamingController.deprecationLogger.warn).toHaveBeenCalledWith(
      'Using deprecated fileStreamingController.handleStream method',
      expect.objectContaining({
        requestPath: '/stream',
        migration: 'Use fileHostingController.getFile instead',
      }),
    );
  });

  test('delegates to fileHostingController.getFile', async () => {
    req.query.file = 'test.mp4';
    await fileStreamingController.handleStream(req, sock);
    expect(fileHostingController.getFile).toHaveBeenCalledWith(req, sock);
  });
});
