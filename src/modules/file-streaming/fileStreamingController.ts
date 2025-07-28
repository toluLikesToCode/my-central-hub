import { Socket } from 'net';
import { IncomingRequest } from '../../entities/http';
import logger from '../../utils/logger';
import { formatDate } from '../../utils/dateFormatter';
import { fileHostingController } from '../file-hosting/fileHostingController';

// Create a specific deprecation logger
const deprecationLogger = logger.child({
  module: 'file-streaming',
  deprecation: true,
});

/**
 * @deprecated This controller is deprecated and will be removed in the next major version.
 * Please use the fileHostingController.getFile() method from the file-hosting module instead.
 *
 * FileStreamingController
 *
 * Handles streaming media files with support for HTTP Range requests
 * to enable efficient video/audio streaming with seeking capabilities.
 */
export const fileStreamingController = {
  /**
   * @deprecated This method is deprecated and will be removed in the next major version.
   * Please use fileHostingController.getFile() instead.
   *
   * Streams a file from the media directory with Range header support
   *
   * This handler provides optimized media file delivery by supporting:
   * - Partial content (206) responses for HTTP Range requests
   * - Proper MIME type detection for various media formats
   * - Efficient byte-range serving for video/audio seeking
   * - Appropriate error handling with meaningful status codes
   *
   * @param req - The incoming HTTP request with file query parameter
   * @param sock - The TCP socket to write response to
   * @returns {Promise<void>} - Resolves when streaming is complete or on error
   */
  async handleStream(req: IncomingRequest, sock: Socket): Promise<void> {
    // Log deprecation warning for monitoring
    deprecationLogger.warn('Using deprecated fileStreamingController.handleStream method', {
      requestPath: req.path,
      clientIp: req.headers['x-forwarded-for'] || sock.remoteAddress,
      timestamp: formatDate(new Date()),
      migration: 'Use fileHostingController.getFile instead',
    });

    // Simply forward to the file-hosting controller to ensure compatibility
    return fileHostingController.getFile(req, sock);
  },
};

/**
 * @module fileStreamingController
 * @deprecated This module is deprecated and will be removed in the next major version.
 * Please use the file-hosting module instead.
 *
 * @description This module handles streaming media files with support for HTTP Range requests.
 * It provides optimized media file delivery by supporting partial content (206) responses for HTTP Range requests,
 * proper MIME type detection for various media formats, efficient byte-range serving for video/audio seeking,
 * and appropriate error handling with meaningful status codes.
 *
 * @version 1.0.1
 * @date 2025-05-06
 */
