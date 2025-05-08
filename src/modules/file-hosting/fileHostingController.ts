// src/modules/file-hosting/fileHostingController.ts
import { Socket } from 'net';
import { sendResponse } from '../../entities/sendResponse';
import { IncomingRequest } from '../../entities/http';
import { FileHostingService } from './fileHostingService';
import { getHeader, getQuery } from '../../utils/httpHelpers';
import { config } from '../../config/server.config';
import logger from '../../utils/logger';
import { getMimeType } from '../../utils/helpers';
import { Readable } from 'stream';
import { formatDate } from '../../utils/dateFormatter';

// Create module-specific logger with metadata
const fileLogger = logger.child({
  module: 'file-hosting',
  component: 'controller',
  feature: 'media-files',
});

const fileSvc = new FileHostingService(config.mediaDir);

/**
 * FileHostingController
 *
 * Provides a complete API for managing media files in the system:
 * - File retrieval with range support
 * - File listing
 * - File upload
 * - File deletion
 *
 * All operations include appropriate logging, error handling, and
 * security validations for file operations.
 */
export const fileHostingController = {
  /**
   * Serves a specific file with Range header support
   *
   * @param req - The incoming HTTP request with file query parameter
   * @param sock - The TCP socket to write the response to
   * @returns {Promise<void>} Resolves when file serving completes
   */
  async getFile(req: IncomingRequest, sock: Socket): Promise<void> {
    const requestStart = Date.now();
    const requestInfo = {
      url: req.url,
      path: req.path,
      query: req.query,
      requestTime: formatDate(new Date()),
      clientIp: req.headers['x-forwarded-for'] || sock.remoteAddress,
      userAgent: req.headers['user-agent'] || 'unknown',
    };

    // Check if socket is still connected
    if (sock.destroyed) {
      fileLogger.warn('Socket closed before processing file request', {
        remoteAddress: sock.remoteAddress,
      });
      return;
    }

    const fileName = getQuery(req, 'file');
    if (!fileName) {
      fileLogger.warn('Missing file parameter in request', requestInfo);
      sendResponse(
        sock,
        400,
        { 'Content-Type': 'text/plain', Connection: 'close' },
        'Missing required "file" query parameter.',
      );
      return;
    }

    // Extract and normalize headers for caching optimizations
    const ifNoneMatchHeader = getHeader(req, 'if-none-match');
    const ifModifiedSinceHeader = getHeader(req, 'if-modified-since');
    // These compression-related variables are reserved for future implementation
    // const acceptEncodingHeader = getHeader(req, 'accept-encoding') || '';
    // const supportsGzip = acceptEncodingHeader.includes('gzip');
    // const supportsBrotli = acceptEncodingHeader.includes('br');

    try {
      const rangeHdr = getHeader(req, 'range');
      let stream: Readable;

      // Add error handling for file stat to give clear errors
      let fileStat;
      try {
        fileStat = await fileSvc.stat(fileName);
      } catch (statErr) {
        fileLogger.error(`Failed to get file stats for "${fileName}"`, {
          error: (statErr as Error).message,
          path: `${config.mediaDir}/${fileName}`,
        });

        if (!sock.destroyed) {
          sendResponse(
            sock,
            404,
            { 'Content-Type': 'text/plain', Connection: 'close' },
            `File "${fileName}" not found or inaccessible.`,
          );
        }
        return;
      }

      const size = fileStat.size;
      const mimeType = getMimeType(fileName) || 'application/octet-stream';
      const isBinaryContent =
        mimeType.includes('image/') ||
        mimeType.includes('video/') ||
        mimeType.includes('audio/') ||
        mimeType.includes('application/octet-stream');

      // Generate ETag based on file size and modification time for caching
      const lastModified = fileStat.mtime.toUTCString();
      const etag = `"${fileStat.size}-${fileStat.mtime.getTime()}"`;

      // Check for conditional requests - If-None-Match and If-Modified-Since
      if (
        ifNoneMatchHeader === etag ||
        (ifModifiedSinceHeader && new Date(ifModifiedSinceHeader) >= fileStat.mtime)
      ) {
        // Client already has the latest version
        fileLogger.debug('File not modified, returning 304', {
          fileName,
          etag,
          lastModified,
          ifNoneMatch: ifNoneMatchHeader,
          ifModifiedSince: ifModifiedSinceHeader,
        });

        sendResponse(
          sock,
          304,
          {
            ETag: etag,
            'Last-Modified': lastModified,
            'Cache-Control': 'public, max-age=86400',
          },
          undefined, // No body for 304 responses
        );
        return;
      }

      fileLogger.debug('File metadata retrieved', {
        fileName,
        size: `${(size / 1024).toFixed(2)} KB`,
        mimeType,
        isBinary: isBinaryContent,
        lastModified: formatDate(fileStat.mtime),
        hasRangeHeader: !!rangeHdr,
        requestedBy: requestInfo.clientIp,
        userAgent: requestInfo.userAgent,
      });

      // Check again if socket is still connected
      if (sock.destroyed) {
        fileLogger.warn('Socket closed during file retrieval preparation', {
          fileName,
          remoteAddress: sock.remoteAddress,
        });
        return;
      }

      // Store range match for later reference if needed
      let rangeMatch: RegExpExecArray | null = null;

      if (rangeHdr) {
        rangeMatch = /bytes=(\d*)-(\d*)/.exec(rangeHdr);
        if (!rangeMatch) {
          fileLogger.warn('Invalid range header format', {
            rangeHeader: rangeHdr,
            fileName,
          });

          sendResponse(
            sock,
            416,
            { 'Content-Type': 'text/plain', Connection: 'close' },
            '416 Range Not Satisfiable',
          );
          return;
        }

        const startStr = rangeMatch[1];
        const endStr = rangeMatch[2];
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
          fileLogger.warn('Range out of bounds', {
            rangeHeader: rangeHdr,
            fileSize: size,
            requestedStart: start,
            requestedEnd: end,
          });

          sendResponse(
            sock,
            416,
            { 'Content-Type': 'text/plain', Connection: 'close' },
            '416 Range Not Satisfiable',
          );
          return;
        }

        try {
          // Use the caching service for range requests too
          stream = await fileSvc.readFile(fileName, { start, end });
          if (!stream) throw new Error('Stream is undefined');
        } catch (streamErr) {
          fileLogger.error(`Failed to create file stream for "${fileName}"`, {
            error: (streamErr as Error).message,
            range: `${start}-${end}`,
          });

          if (!sock.destroyed) {
            sendResponse(
              sock,
              500,
              { 'Content-Type': 'text/plain', Connection: 'close' },
              'Error creating file stream',
            );
          }
          return;
        }

        // Improved stream error handling with explicit stream destruction
        stream.on('error', (err: Error) => {
          fileLogger.error(`Stream error for "${fileName}"`, {
            error: err.message,
            stack: err.stack,
            range: `${start}-${end}`,
          });

          // Destroy stream to prevent memory leaks
          if (!stream.destroyed) {
            stream.destroy();
          }

          if (!sock.destroyed) {
            try {
              sendResponse(
                sock,
                500,
                { 'Content-Type': 'text/plain', Connection: 'close' },
                'Internal Server Error',
              );
            } catch (sendErr) {
              fileLogger.error('Error sending error response', {
                error: (sendErr as Error).message,
              });
              if (!sock.destroyed) sock.end();
            }
          }
        });

        const len = end - start + 1;
        fileLogger.debug('Serving partial content', {
          start,
          end,
          bytes: len,
          percentage: `${Math.round((len / size) * 100)}%`,
          isBinary: isBinaryContent,
        });

        // Final socket check before sending response
        if (sock.destroyed) {
          fileLogger.warn('Socket closed before sending partial content', {
            fileName,
            range: `${start}-${end}`,
          });
          if (!stream.destroyed) {
            stream.destroy();
          }
          return;
        }

        // Add headers based on content type and other factors
        const responseHeaders: Record<string, string> = {
          'Content-Type': mimeType,
          'Accept-Ranges': 'bytes',
          'Content-Range': `bytes ${start}-${end}/${size}`,
          'Content-Length': String(len),
          'Cache-Control': 'public, max-age=86400, stale-while-revalidate=43200',
          ETag: etag,
          'Last-Modified': lastModified,
          // Performance headers
          'X-Content-Type-Options': 'nosniff',
          'Timing-Allow-Origin': '*',
        };

        // Set connection header for large files
        if (isBinaryContent && size > 1024 * 1024) {
          responseHeaders['Connection'] = 'close';
        }

        try {
          sendResponse(sock, 206, responseHeaders, stream);
        } catch (sendErr) {
          fileLogger.error('Error sending partial content response', {
            error: (sendErr as Error).message,
            fileName,
            range: `${start}-${end}`,
          });
          if (!stream.destroyed) {
            stream.destroy();
          }
        }
      } else {
        try {
          // Get the file through the caching service (now faster with our improvements)
          stream = await fileSvc.readFile(fileName);
          if (!stream) throw new Error('Stream is undefined');
        } catch (streamErr) {
          fileLogger.error(`Failed to create file stream for "${fileName}"`, {
            error: (streamErr as Error).message,
          });

          if (!sock.destroyed) {
            sendResponse(
              sock,
              500,
              { 'Content-Type': 'text/plain', Connection: 'close' },
              'Error creating file stream',
            );
          }
          return;
        }

        // Improved stream error handling with explicit stream destruction
        stream.on('error', (err: Error) => {
          fileLogger.error(`Stream error for "${fileName}"`, {
            error: err.message,
            stack: err.stack,
          });

          // Destroy stream to prevent memory leaks
          if (!stream.destroyed) {
            stream.destroy();
          }

          if (!sock.destroyed) {
            try {
              sendResponse(
                sock,
                500,
                { 'Content-Type': 'text/plain', Connection: 'close' },
                'Internal Server Error',
              );
            } catch (sendErr) {
              fileLogger.error('Error sending error response', {
                error: (sendErr as Error).message,
              });
              if (!sock.destroyed) sock.end();
            }
          }
        });

        fileLogger.debug('Serving full content', {
          mimeType,
          size,
          isBinary: isBinaryContent,
          fromCache: false, // Log if it came from cache or not (handled in service)
        });

        // Final socket check before sending response
        if (sock.destroyed) {
          fileLogger.warn('Socket closed before sending full content', {
            fileName,
          });
          if (!stream.destroyed) {
            stream.destroy();
          }
          return;
        }

        // Enhanced headers for better performance and caching
        const responseHeaders: Record<string, string> = {
          'Content-Type': mimeType,
          'Content-Length': String(size),
          'Cache-Control': 'public, max-age=86400, stale-while-revalidate=43200',
          'Accept-Ranges': 'bytes',
          ETag: etag,
          'Last-Modified': lastModified,
          // Performance headers
          'X-Content-Type-Options': 'nosniff',
          'Timing-Allow-Origin': '*',
        };

        // For immutable content like images, add immutable flag
        if (isBinaryContent && mimeType.startsWith('image/')) {
          responseHeaders['Cache-Control'] = 'public, max-age=31536000, immutable';
        }

        // Set appropriate connection header for large files
        if (isBinaryContent && size > 1024 * 1024) {
          responseHeaders['Connection'] = 'close';
        }

        try {
          sendResponse(sock, 200, responseHeaders, stream);
        } catch (sendErr) {
          fileLogger.error('Error sending full content response', {
            error: (sendErr as Error).message,
            fileName,
          });
          if (!stream.destroyed) {
            stream.destroy();
          }
        }
      }

      // Add completion handler using socket events
      sock.on('finish', () => {
        const duration = Date.now() - requestStart;
        const rangeInfo = rangeHdr
          ? `partial (${rangeMatch ? rangeMatch[1] + '-' + rangeMatch[2] : 'unknown range'})`
          : `${size} bytes`;

        fileLogger.info(`File served: "${fileName}"`, {
          fileName,
          duration: `${duration}ms`,
          status: rangeHdr ? 206 : 200,
          size: rangeInfo,
          isBinary: isBinaryContent,
        });
      });

      // Add close handler to clean up resources if the client disconnects prematurely
      sock.on('close', () => {
        const duration = Date.now() - requestStart;
        if (!stream.destroyed) {
          stream.destroy();
          fileLogger.warn(`Client disconnected during file transfer: "${fileName}"`, {
            duration: `${duration}ms`,
            isBinary: isBinaryContent,
          });
        }
      });
    } catch (err: unknown) {
      const error = err as Error;
      const duration = Date.now() - requestStart;

      fileLogger.error(`Failed to serve file "${fileName}"`, {
        error: error.message,
        stack: error.stack,
        duration: `${duration}ms`,
        ...requestInfo,
      });

      if (!sock.destroyed) {
        try {
          sendResponse(
            sock,
            404,
            { 'Content-Type': 'text/plain', Connection: 'close' },
            `File "${fileName}" not found.`,
          );
        } catch (sendErr) {
          fileLogger.error('Error sending error response', {
            error: (sendErr as Error).message,
          });
          if (!sock.destroyed) sock.end();
        }
      }
    }
  },

  /**
   * Lists all available files in the media directory
   *
   * @param req - The incoming HTTP request
   * @param sock - The TCP socket to write the response to
   * @returns {Promise<void>} Resolves when file listing completes
   */
  async listFiles(req: IncomingRequest, sock: Socket): Promise<void> {
    const requestStart = Date.now();

    fileLogger.info('File listing requested', {
      remoteAddress: sock.remoteAddress,
      url: req.url,
      timestamp: formatDate(new Date()),
    });

    try {
      // Check if socket is still connected before proceeding
      if (sock.destroyed) {
        fileLogger.warn('Socket already closed, aborting file listing operation', {
          remoteAddress: sock.remoteAddress,
        });
        return;
      }

      const files = await fileSvc.listFiles();

      // Enhance response with file metadata
      const fileDetails = await Promise.all(
        files.map(async (fileName) => {
          try {
            const stats = await fileSvc.stat(fileName);
            return {
              name: fileName,
              size: stats.size,
              mimeType: getMimeType(fileName) || 'application/octet-stream',
              lastModified: formatDate(stats.mtime),
            };
          } catch (statErr) {
            // Don't throw errors for individual files, just report the issue
            fileLogger.warn(`Could not read stats for file: ${fileName}`, {
              error: (statErr as Error).message,
            });
            return { name: fileName, error: 'Could not read file info' };
          }
        }),
      );

      const duration = Date.now() - requestStart;

      fileLogger.success('File listing completed', {
        fileCount: files.length,
        duration: `${duration}ms`,
      });

      // Check again if socket is still connected before sending response
      if (sock.destroyed) {
        fileLogger.warn('Socket disconnected during file listing operation', {
          remoteAddress: sock.remoteAddress,
          duration: `${duration}ms`,
        });
        return;
      }

      // Use try-catch to handle potential socket errors during response
      try {
        sendResponse(
          sock,
          200,
          {
            'Content-Type': 'application/json',
            Connection: 'close', // Explicitly tell client we'll close the connection
          },
          JSON.stringify(fileDetails),
        );
      } catch (sendErr) {
        fileLogger.error('Error sending file listing response', {
          error: (sendErr as Error).message,
          stack: (sendErr as Error).stack,
        });
      }
    } catch (err: unknown) {
      const error = err as Error;
      const duration = Date.now() - requestStart;

      fileLogger.error('Failed to list files', {
        error: error.message,
        stack: error.stack,
        duration: `${duration}ms`,
      });

      // Check if socket is still connected before sending error response
      if (!sock.destroyed) {
        try {
          sendResponse(
            sock,
            500,
            {
              'Content-Type': 'text/plain',
              Connection: 'close',
            },
            'Server error: Unable to list files',
          );
        } catch (sendErr) {
          fileLogger.error('Error sending error response', {
            error: (sendErr as Error).message,
          });
        }
      }
    }
  },

  /**
   * Uploads a new media file to the server
   *
   * @param req - The incoming HTTP request with file data in body
   * @param sock - The TCP socket to write the response to
   * @returns {Promise<void>} Resolves when file upload completes
   */
  async uploadFile(req: IncomingRequest, sock: Socket): Promise<void> {
    const requestStart = Date.now();
    const requestInfo = {
      url: req.url,
      headers: req.headers,
      contentType: req.headers['content-type'],
      contentLength: req.headers['content-length'],
      timestamp: formatDate(new Date()),
      clientIp: req.headers['x-forwarded-for'] || sock.remoteAddress,
    };

    fileLogger.info('File upload requested', requestInfo);

    // Check if socket is still connected
    if (sock.destroyed) {
      fileLogger.warn('Socket closed before processing upload request', {
        remoteAddress: sock.remoteAddress,
      });
      return;
    }

    // Assume fileName is provided as a query param or header, and body is the file data
    const fileName = getQuery(req, 'file') || req.headers['x-filename'];
    if (!fileName) {
      fileLogger.warn('Missing filename in upload request', requestInfo);
      sendResponse(
        sock,
        400,
        { 'Content-Type': 'text/plain', Connection: 'close' },
        'Missing file name',
      );
      return;
    }

    // Get MIME type safely with fallback to empty string
    const mimeType = getMimeType(fileName) || '';

    // Validate MIME type - only allow media files
    if (
      !mimeType.startsWith('image/') &&
      !mimeType.startsWith('video/') &&
      !mimeType.startsWith('audio/')
    ) {
      fileLogger.warn('Invalid file type rejected', {
        fileName,
        mimeType: mimeType || 'unknown',
        allowed: ['image/*', 'video/*', 'audio/*'],
      });

      sendResponse(
        sock,
        400,
        { 'Content-Type': 'text/plain', Connection: 'close' },
        'Only media files allowed',
      );
      return;
    }

    // Improved body validation with detailed error
    if (!req.body) {
      fileLogger.warn('Empty file upload attempt', {
        fileName,
        ...requestInfo,
      });

      sendResponse(
        sock,
        400,
        { 'Content-Type': 'text/plain', Connection: 'close' },
        'Binary file load error: invalid or missing file source',
      );
      return;
    }

    try {
      // Check if socket is still connected before proceeding with file processing
      if (sock.destroyed) {
        fileLogger.warn('Socket closed before file processing', {
          fileName,
          remoteAddress: sock.remoteAddress,
        });
        return;
      }

      // Improved buffer handling with better error logging
      let data: Buffer;
      try {
        if (typeof req.body === 'string') {
          data = Buffer.from(req.body);
        } else if (Buffer.isBuffer(req.body)) {
          data = req.body;
        } else {
          throw new Error('Unsupported body type: ' + typeof req.body);
        }
      } catch (bufferErr) {
        fileLogger.error('Failed to process request body', {
          error: (bufferErr as Error).message,
          contentType: req.headers['content-type'],
          bodyType: typeof req.body,
        });
        sendResponse(
          sock,
          400,
          { 'Content-Type': 'text/plain', Connection: 'close' },
          'Binary file load error: invalid or missing file source',
        );
        return;
      }

      const fileSize = data.length;

      // Additional validation for empty files
      if (fileSize === 0) {
        fileLogger.warn('Zero-length file upload attempted', { fileName });
        sendResponse(
          sock,
          400,
          { 'Content-Type': 'text/plain', Connection: 'close' },
          'Cannot upload empty file',
        );
        return;
      }

      fileLogger.debug('Processing file upload', {
        fileName,
        size: `${(fileSize / 1024).toFixed(2)} KB`,
        mimeType,
      });

      // Convert Buffer to async iterable with chunking for large files
      async function* bufferToAsyncIterable(buffer: Buffer) {
        // For large files, chunk them to avoid memory issues
        const CHUNK_SIZE = 1024 * 1024; // 1MB chunks
        if (buffer.length > CHUNK_SIZE) {
          for (let i = 0; i < buffer.length; i += CHUNK_SIZE) {
            yield buffer.slice(i, Math.min(i + CHUNK_SIZE, buffer.length));
          }
        } else {
          yield buffer;
        }
      }

      await fileSvc.saveFile(fileName, bufferToAsyncIterable(data));

      const duration = Date.now() - requestStart;
      fileLogger.success(`File "${fileName}" uploaded successfully`, {
        size: fileSize,
        mimeType,
        duration: `${duration}ms`,
        storagePath: `${config.mediaDir}/${fileName}`,
      });

      // Check if socket is still connected before sending response
      if (sock.destroyed) {
        fileLogger.warn('Socket closed after upload completed but before response', {
          fileName,
          remoteAddress: sock.remoteAddress,
        });
        return;
      }

      try {
        sendResponse(
          sock,
          200,
          {
            'Content-Type': 'application/json',
            Connection: 'close',
          },
          JSON.stringify({
            success: true,
            fileName,
            size: fileSize,
            mimeType,
            message: 'Upload successful',
          }),
        );
      } catch (sendErr) {
        fileLogger.error('Error sending upload success response', {
          error: (sendErr as Error).message,
          fileName,
        });
      }
    } catch (err: unknown) {
      const error = err as Error;
      const duration = Date.now() - requestStart;

      fileLogger.error(`File upload failed for "${fileName}"`, {
        error: error.message,
        stack: error.stack,
        duration: `${duration}ms`,
        ...requestInfo,
      });

      // Check if socket is still connected before sending error response
      if (!sock.destroyed) {
        try {
          sendResponse(
            sock,
            500,
            { 'Content-Type': 'text/plain', Connection: 'close' },
            `Upload failed: ${error.message}`,
          );
        } catch (sendErr) {
          fileLogger.error('Error sending upload failure response', {
            error: (sendErr as Error).message,
          });
        }
      }
    }
  },

  /**
   * Deletes a media file from the server
   *
   * @param req - The incoming HTTP request with file query parameter
   * @param sock - The TCP socket to write the response to
   * @returns {Promise<void>} Resolves when file deletion completes
   */
  async deleteFile(req: IncomingRequest, sock: Socket): Promise<void> {
    const requestStart = Date.now();
    const requestInfo = {
      url: req.url,
      query: req.query,
      timestamp: formatDate(new Date()),
      clientIp: req.headers['x-forwarded-for'] || sock.remoteAddress,
    };

    fileLogger.info('File deletion requested', requestInfo);

    // Check if socket is still connected
    if (sock.destroyed) {
      fileLogger.warn('Socket closed before processing delete request', {
        remoteAddress: sock.remoteAddress,
      });
      return;
    }

    const fileName = getQuery(req, 'file');
    if (!fileName) {
      fileLogger.warn('Missing filename in delete request', requestInfo);
      sendResponse(
        sock,
        400,
        { 'Content-Type': 'text/plain', Connection: 'close' },
        'Missing file name',
      );
      return;
    }

    try {
      // Get file info before deletion for logging
      let fileInfo = {};
      try {
        const stats = await fileSvc.stat(fileName);
        fileInfo = {
          size: stats.size,
          lastModified: formatDate(stats.mtime),
        };
      } catch (statErr) {
        fileLogger.warn(`Could not get file stats before deletion`, {
          fileName,
          error: (statErr as Error).message,
        });
      }

      // Check if socket is still connected before proceeding with deletion
      if (sock.destroyed) {
        fileLogger.warn('Socket closed before file deletion', {
          fileName,
          remoteAddress: sock.remoteAddress,
        });
        return;
      }

      // Use the service's deleteFile method for safer deletion
      await fileSvc.deleteFile(fileName);

      const duration = Date.now() - requestStart;
      fileLogger.success(`File "${fileName}" deleted successfully`, {
        ...fileInfo,
        duration: `${duration}ms`,
      });

      // Check if socket is still connected before sending response
      if (sock.destroyed) {
        fileLogger.warn('Socket closed after deletion but before response', {
          fileName,
          remoteAddress: sock.remoteAddress,
        });
        return;
      }

      try {
        sendResponse(
          sock,
          200,
          {
            'Content-Type': 'application/json',
            Connection: 'close',
          },
          JSON.stringify({
            success: true,
            fileName,
            message: 'File deleted',
          }),
        );
      } catch (sendErr) {
        fileLogger.error('Error sending delete success response', {
          error: (sendErr as Error).message,
          fileName,
        });
      }
    } catch (err: unknown) {
      const error = err as Error;
      const duration = Date.now() - requestStart;

      fileLogger.error(`Failed to delete file "${fileName}"`, {
        error: error.message,
        stack: error.stack,
        duration: `${duration}ms`,
        ...requestInfo,
      });

      // Check if socket is still connected before sending error response
      if (!sock.destroyed) {
        try {
          sendResponse(
            sock,
            404,
            { 'Content-Type': 'text/plain', Connection: 'close' },
            'File not found or could not be deleted',
          );
        } catch (sendErr) {
          fileLogger.error('Error sending delete failure response', {
            error: (sendErr as Error).message,
          });
        }
      }
    }
  },
};
