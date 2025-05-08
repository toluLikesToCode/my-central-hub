import { Socket } from 'net';
import { sendResponse } from '../../entities/sendResponse';
import { IncomingRequest } from '../../entities/http';
import { getHeader } from '../../utils/httpHelpers';
import { config } from '../../config/server.config';
import logger from '../../utils/logger';
import { getMimeType } from '../../utils/helpers';
import { Readable } from 'stream';
import { FileHostingService } from '../file-hosting/fileHostingService';
import { formatDate } from '../../utils/dateFormatter';

const fileSvc = new FileHostingService(config.mediaDir);

/**
 * FileStreamingController
 *
 * Handles streaming media files with support for HTTP Range requests
 * to enable efficient video/audio streaming with seeking capabilities.
 */
export const fileStreamingController = {
  /**
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
    const requestStart = Date.now();
    const requestMeta = {
      url: req.url,
      path: req.path,
      query: req.query,
      headers: req.headers,
      method: req.method,
      requestTime: formatDate(new Date()),
    };

    if (!config.testMode) {
      logger.info(`Stream request received for ${req.path}`, requestMeta);
    }

    const fileName = req.query.file as string | undefined;
    if (!fileName) {
      logger.warn('Missing file parameter in stream request', requestMeta);
      sendResponse(
        sock,
        400,
        { 'Content-Type': 'text/plain' },
        'Missing required "file" query parameter.',
      );
      return;
    }

    try {
      const rangeHdr = getHeader(req, 'range');
      const fileStat = await fileSvc.stat(fileName);
      const size = fileStat.size;
      // Use getMimeType with a fallback for better error handling
      const mimeType = getMimeType(fileName) || 'application/octet-stream';
      let stream: Readable;

      // Log file metadata
      logger.debug('File metadata retrieved', {
        fileName,
        size,
        mimeType,
        lastModified: formatDate(fileStat.mtime),
        hasRangeHeader: !!rangeHdr,
      });

      if (rangeHdr) {
        const m = /bytes=(\d*)-(\d*)/.exec(rangeHdr);
        if (!m) {
          logger.warn('Invalid range header format', { rangeHeader: rangeHdr, fileName });
          sendResponse(sock, 416, { 'Content-Type': 'text/plain' }, '416 Range Not Satisfiable');
          sock.end();
          return;
        }

        const startStr = m[1];
        const endStr = m[2];
        let start: number;
        let end: number;

        if (startStr) {
          start = parseInt(startStr, 10);
          end = endStr ? parseInt(endStr, 10) : size - 1;
        } else {
          const suffix = parseInt(endStr, 10);
          start = size - suffix;
          end = size - 1;
        }

        if (start > end || start < 0 || end >= size) {
          logger.warn('Range out of bounds', {
            rangeHeader: rangeHdr,
            fileSize: size,
            requestedStart: start,
            requestedEnd: end,
          });
          sendResponse(sock, 416, { 'Content-Type': 'text/plain' }, '416 Range Not Satisfiable');
          sock.end();
          return;
        }

        stream = await fileSvc.readFile(fileName, { start, end });
        if (!stream) throw new Error('Stream is undefined');

        const len = end - start + 1;
        logger.debug('Serving partial content', {
          start,
          end,
          length: len,
          percentage: `${Math.round((len / size) * 100)}%`,
        });

        sendResponse(
          sock,
          206,
          {
            'Content-Type': mimeType,
            'Accept-Ranges': 'bytes',
            'Content-Range': `bytes ${start}-${end}/${size}`,
            'Content-Length': String(len),
          },
          stream,
        );
      } else {
        stream = await fileSvc.readFile(fileName);
        if (!stream) throw new Error('Stream is undefined');

        logger.debug('Serving full content', {
          size,
          mimeType,
        });

        sendResponse(
          sock,
          200,
          { 'Content-Type': mimeType, 'Content-Length': String(size) },
          stream,
        );
      }

      // Add stream error handler
      stream.on('error', (err: Error) => {
        logger.error(`Stream error occurred for "${fileName}"`, {
          error: err.message,
          stack: err.stack,
        });
        try {
          sendResponse(sock, 500, { 'Content-Type': 'text/plain' }, 'Error while streaming file');
        } catch (sendErr) {
          logger.error('Failed to send error response', { error: sendErr });
        }
      });

      // Add stream finish handler
      stream.on('end', () => {
        const duration = Date.now() - requestStart;
        logger.debug('Stream completed', {
          fileName,
          duration: `${duration}ms`,
          status: rangeHdr ? 206 : 200,
        });
      });
    } catch (err) {
      const error = err as Error;
      const duration = Date.now() - requestStart;

      logger.error(`Failed to stream file "${fileName}"`, {
        error: error.message,
        stack: error.stack,
        duration: `${duration}ms`,
        ...requestMeta,
      });

      sendResponse(sock, 404, { 'Content-Type': 'text/plain' }, `File "${fileName}" not found.`);
    }
  },
};
/**
 * @module fileStreamingController
 * @description This module handles streaming media files with support for HTTP Range requests.
 * It provides optimized media file delivery by supporting partial content (206) responses for HTTP Range requests,
 * proper MIME type detection for various media formats, efficient byte-range serving for video/audio seeking,
 * and appropriate error handling with meaningful status codes.
 *
 * @version 1.0.1
 * @date 2025-05-06
 */
