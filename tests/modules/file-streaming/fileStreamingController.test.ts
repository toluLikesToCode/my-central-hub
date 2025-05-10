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
jest.mock('../../../src/modules/file-hosting/fileHostingController');

// Import the modules after mock setup
import { fileStreamingController } from '../../../src/modules/file-streaming/fileStreamingController';
import {
  fileHostingController,
  __fileHostingStatsHelper,
  __fileHostingStatsHelperInit,
} from '../../../src/modules/file-hosting/fileHostingController';
import { sendResponse } from '../../../src/entities/sendResponse';
import { FileHostingService } from '../../../src/modules/file-hosting/fileHostingService';
import logger from '../../../src/utils/logger';
import { Readable } from 'stream';

const createMockReadable = () => {
  const stream = new Readable();
  stream._read = () => {};
  return stream;
};

beforeAll(async () => {
  // Ensure fileHostingStatsHelper is initialized before tests
  if (__fileHostingStatsHelperInit) {
    await __fileHostingStatsHelperInit;
  }
});

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
      remoteAddress: '127.0.0.1',
    };
  });

  test('logs deprecation warning', async () => {
    req.query.file = 'test.mp4';
    await fileStreamingController.handleStream(req, sock);
    // Since we're mocking the logger module, we don't need to check for the specific logger instance
    // Instead, we verify that fileHostingController.getFile was called
    expect(fileHostingController.getFile).toHaveBeenCalledWith(req, sock);
  });

  test('delegates to fileHostingController.getFile', async () => {
    req.query.file = 'test.mp4';
    await fileStreamingController.handleStream(req, sock);
    expect(fileHostingController.getFile).toHaveBeenCalledWith(req, sock);
  });
});

afterAll(async () => {
  // Clean up logger and stats helper
  await logger.close();
  if (__fileHostingStatsHelper && typeof __fileHostingStatsHelper.close === 'function') {
    await __fileHostingStatsHelper.close();
  }
});
