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
// --- NEW: Import zlib for compression ---
import * as zlib from 'zlib';

// Create module-specific logger with metadata
const fileLogger = logger.child({
  module: 'file-hosting',
  component: 'controller',
  feature: 'media-files',
});

// Use staticDir instead of mediaDir for file hosting
const fileSvc = new FileHostingService(config.staticDir);

// --- NEW: Define compressible MIME types ---
const COMPRESSIBLE_MIME_TYPES = new Set([
  'text/plain',
  'text/html',
  'text/css',
  'application/javascript',
  'application/json',
  'application/xml',
  'image/svg+xml',
]);

function isCompressible(mimeType: string): boolean {
  return COMPRESSIBLE_MIME_TYPES.has(mimeType);
}

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
   * Serves a specific file with Range header support and optional compression
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

    // Extract headers
    const ifNoneMatchHeader = getHeader(req, 'if-none-match');
    const ifModifiedSinceHeader = getHeader(req, 'if-modified-since');
    // --- NEW: Process Accept-Encoding ---
    const acceptEncodingHeader = (getHeader(req, 'accept-encoding') || '').toLowerCase();
    const supportsBrotli = /\bbr\b/.test(acceptEncodingHeader);
    const supportsGzip = /\bgzip\b/.test(acceptEncodingHeader);
    const supportsDeflate = /\bdeflate\b/.test(acceptEncodingHeader);
    const rangeHdr = getHeader(req, 'range'); // Keep range header check

    try {
      let fileStat;
      try {
        fileStat = await fileSvc.stat(fileName);
      } catch (statErr) {
        fileLogger.error(`Failed to get file stats for "${fileName}"`, {
          error: (statErr as Error).message,
          path: `${config.staticDir}/${fileName}`,
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
      const lastModified = fileStat.mtime.toUTCString();
      const etag = `"${fileStat.size}-${fileStat.mtime.getTime()}"`;

      const isBinaryContent =
        mimeType.includes('image/') ||
        mimeType.includes('video/') ||
        mimeType.includes('audio/') ||
        mimeType.includes('application/octet-stream');

      // --- Handle Conditional Requests (304 Not Modified) ---
      if (
        ifNoneMatchHeader === etag ||
        (ifModifiedSinceHeader && new Date(ifModifiedSinceHeader) >= fileStat.mtime)
      ) {
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

      // --- Handle Range Requests (206 Partial Content) ---
      // --- NOTE: Compression is NOT applied to range requests ---
      if (rangeHdr) {
        const rangeMatch = /bytes=(\d*)-(\d*)/.exec(rangeHdr);
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

        let start: number;
        let end: number;
        const startStr = rangeMatch[1];
        const endStr = rangeMatch[2];

        if (startStr) {
          start = parseInt(startStr, 10);
          end = endStr ? parseInt(endStr, 10) : size - 1;
        } else {
          // Handle suffix range like bytes=-500
          const suffix = parseInt(endStr, 10);
          if (isNaN(suffix) || suffix <= 0) {
            fileLogger.warn('Invalid suffix range header format', {
              rangeHeader: rangeHdr,
              fileName,
            });
            sendResponse(
              sock,
              416,
              { 'Content-Range': `bytes */${size}` }, // Recommended
              '416 Range Not Satisfiable',
            );
            return;
          }
          start = Math.max(0, size - suffix); // Ensure start is not negative
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
            { 'Content-Range': `bytes */${size}` },
            '416 Range Not Satisfiable',
          );
          return;
        }

        let stream: Readable;
        try {
          // Read only the specified range
          stream = await fileSvc.readFile(fileName, { start, end });
          if (!stream) throw new Error('Stream is undefined');
        } catch (streamErr) {
          fileLogger.error(`Failed to create file stream for "${fileName}"`, {
            error: (streamErr as Error).message,
            range: `${start}-${end}`,
          });
          if (!sock.destroyed)
            sendResponse(
              sock,
              500,
              { 'Content-Type': 'text/plain', Connection: 'close' },
              'Error creating file stream',
            );
          return;
        }

        stream.on('error', (err: Error) => {
          fileLogger.error(`Stream error for "${fileName}"`, {
            error: err.message,
            stack: err.stack,
            range: `${start}-${end}`,
          });
          if (!stream.destroyed) stream.destroy();
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
        fileLogger.debug('Serving partial content (uncompressed)', {
          start,
          end,
          bytes: len,
          percentage: `${Math.round((len / size) * 100)}%`,
          isBinary: isBinaryContent,
        });

        if (sock.destroyed) {
          fileLogger.warn('Socket closed before sending partial content', {
            fileName,
            range: `${start}-${end}`,
          });
          if (!stream.destroyed) stream.destroy();
          return;
        }

        const responseHeaders: Record<string, string> = {
          'Content-Type': mimeType,
          'Accept-Ranges': 'bytes', // Still advertise support
          'Content-Range': `bytes ${start}-${end}/${size}`,
          'Content-Length': String(len),
          'Cache-Control': 'public, max-age=86400, stale-while-revalidate=43200',
          ETag: etag,
          'Last-Modified': lastModified,
          'X-Content-Type-Options': 'nosniff',
          'Timing-Allow-Origin': '*',
          // --- Do NOT add Content-Encoding or Vary for 206 responses ---
        };

        try {
          sendResponse(sock, 206, responseHeaders, stream);
        } catch (sendErr) {
          fileLogger.error('Error sending partial content response', {
            error: (sendErr as Error).message,
            fileName,
            range: `${start}-${end}`,
          });
          if (!stream.destroyed) stream.destroy();
        }

        // Add completion handler using socket events
        sock.on('finish', () => {
          const duration = Date.now() - requestStart;
          fileLogger.info(`Partial file served: "${fileName}"`, {
            fileName,
            duration: `${duration}ms`,
            status: 206,
            range: `${start}-${end}/${size}`,
          });
        });

        sock.on('close', () => {
          if (!stream.destroyed) {
            stream.destroy();
            fileLogger.warn(`Client disconnected during partial transfer: "${fileName}"`, {
              duration: `${Date.now() - requestStart}ms`,
              range: `${start}-${end}`,
            });
          }
        });

        return; // Exit after handling range request
      }

      // --- Handle Full File Requests (200 OK) with potential Compression ---
      let responseStream: Readable;
      let sourceStream: Readable;
      const responseHeaders: Record<string, string> = {
        'Content-Type': mimeType,
        'Accept-Ranges': 'bytes', // Advertise range support
        'Cache-Control': 'public, max-age=86400, stale-while-revalidate=43200',
        ETag: etag,
        'Last-Modified': lastModified,
        'X-Content-Type-Options': 'nosniff',
        'Timing-Allow-Origin': '*',
        // Connection: 'close' for large files logic can remain or be adjusted
      };

      // For immutable content like images, add immutable flag (keeping existing logic)
      if (isBinaryContent && mimeType.startsWith('image/')) {
        responseHeaders['Cache-Control'] = 'public, max-age=31536000, immutable';
      }

      // --- NEW: Compression Logic ---
      let compressionEncoding: string | null = null;
      if (isCompressible(mimeType)) {
        if (supportsBrotli) compressionEncoding = 'br';
        else if (supportsGzip) compressionEncoding = 'gzip';
        else if (supportsDeflate) compressionEncoding = 'deflate';
      }

      try {
        sourceStream = await fileSvc.readFile(fileName); // Get the source stream
        if (!sourceStream) throw new Error('Source stream is undefined');

        if (compressionEncoding) {
          let compressStream: zlib.BrotliCompress | zlib.Gzip | zlib.Deflate;
          if (compressionEncoding === 'br') {
            compressStream = zlib.createBrotliCompress();
          } else if (compressionEncoding === 'gzip') {
            compressStream = zlib.createGzip();
          } else {
            // deflate
            compressStream = zlib.createDeflate();
          }

          // Pipe source through compression
          responseStream = sourceStream.pipe(compressStream);

          // Set compression headers and remove Content-Length
          responseHeaders['Content-Encoding'] = compressionEncoding;
          responseHeaders['Vary'] = 'Accept-Encoding'; // Crucial for caching
          // DO NOT set Content-Length for compressed streams

          fileLogger.debug(`Serving compressed content (${compressionEncoding})`, {
            fileName,
            mimeType,
            originalSize: size,
          });

          // Handle errors on the compression stream as well
          compressStream.on('error', (err: Error) => {
            fileLogger.error(
              `Compression stream error for "${fileName}" (${compressionEncoding})`,
              {
                error: err.message,
                stack: err.stack,
              },
            );
            if (!sourceStream.destroyed) sourceStream.destroy(); // Destroy source if compression fails
            if (!sock.destroyed) sock.destroy(err); // Destroy socket on compression error
          });
        } else {
          // No compression: use original stream and set Content-Length
          responseStream = sourceStream;
          responseHeaders['Content-Length'] = String(size);
          fileLogger.debug('Serving full content (uncompressed)', {
            fileName,
            mimeType,
            size,
          });
        }
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

      // Handle errors on the source stream
      sourceStream.on('error', (err: Error) => {
        fileLogger.error(`Source stream error for "${fileName}"`, {
          error: err.message,
          stack: err.stack,
        });
        // If responseStream is different (i.e., compressed), ensure it's also handled/destroyed if needed.
        // The pipe might handle this, but explicit destruction on source error can be safer.
        if (responseStream !== sourceStream && !responseStream.destroyed) {
          responseStream.destroy(err);
        }
        if (!sock.destroyed) {
          // Don't try to send 500 if headers might already be sent
          sock.destroy(err);
        }
      });

      // Final socket check
      if (sock.destroyed) {
        fileLogger.warn('Socket closed before sending full content', { fileName });
        if (responseStream && !responseStream.destroyed) responseStream.destroy();
        if (sourceStream && responseStream !== sourceStream && !sourceStream.destroyed)
          sourceStream.destroy();
        return;
      }

      // Set appropriate connection header for large files (keeping existing logic)
      if (isBinaryContent && size > 1024 * 1024) {
        responseHeaders['Connection'] = 'close';
      }

      // Send the response (either compressed or original stream)
      try {
        sendResponse(sock, 200, responseHeaders, responseStream);
      } catch (sendErr) {
        fileLogger.error('Error sending full content response', {
          error: (sendErr as Error).message,
          fileName,
        });
        if (!responseStream.destroyed) responseStream.destroy();
        if (sourceStream && responseStream !== sourceStream && !sourceStream.destroyed)
          sourceStream.destroy();
      }

      // Add completion handler using socket events
      sock.on('finish', () => {
        const duration = Date.now() - requestStart;
        fileLogger.info(`Full file served: "${fileName}"`, {
          fileName,
          duration: `${duration}ms`,
          status: 200,
          size: compressionEncoding ? `compressed (${compressionEncoding})` : `${size} bytes`,
        });
      });

      sock.on('close', () => {
        // Ensure streams are destroyed if client disconnects prematurely
        if (responseStream && !responseStream.destroyed) {
          responseStream.destroy();
          fileLogger.warn(`Client disconnected during full transfer: "${fileName}"`, {
            duration: `${Date.now() - requestStart}ms`,
            encoding: compressionEncoding || 'none',
          });
        }
        // Ensure source stream is also cleaned up if it wasn't the response stream
        if (sourceStream && responseStream !== sourceStream && !sourceStream.destroyed) {
          sourceStream.destroy();
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
   * Lists available files with pagination, sorting, and filtering
   *
   * @param req - The incoming HTTP request with optional pagination parameters
   * @param sock - The TCP socket to write the response to
   * @returns {Promise<void>} Resolves when file listing completes
   */
  async listFiles(req: IncomingRequest, sock: Socket): Promise<void> {
    const requestStart = Date.now();

    // Extract pagination, sorting and filtering parameters
    const page = parseInt(req.query?.page as string) || 1;
    const limit = parseInt(req.query?.limit as string) || 20;
    const sort = (req.query?.sort as string) || 'name';
    const order = (req.query?.order as string) || 'asc';
    const filterType = req.query?.type as string;
    const search = req.query?.search as string;
    const dateFrom = req.query?.dateFrom as string;
    const dateTo = req.query?.dateTo as string;
    const sizeFrom = parseInt(req.query?.sizeFrom as string) || 0;
    const sizeTo = parseInt(req.query?.sizeTo as string) || 0;

    fileLogger.info('File listing requested', {
      remoteAddress: sock.remoteAddress,
      url: req.url,
      timestamp: formatDate(new Date()),
      pagination: { page, limit },
      sorting: { sort, order },
      filtering: {
        filterType,
        search,
        dateRange: dateFrom || dateTo ? { from: dateFrom, to: dateTo } : null,
        sizeRange: sizeFrom || sizeTo ? { from: sizeFrom, to: sizeTo } : null,
      },
    });

    try {
      // Check if socket is still connected before proceeding
      if (sock.destroyed) {
        fileLogger.warn('Socket already closed, aborting file listing operation', {
          remoteAddress: sock.remoteAddress,
        });
        return;
      }

      // Get all files first
      const allFiles = await fileSvc.listFiles();

      // Process all files with metadata for filtering
      const filesWithMetadata = await Promise.all(
        allFiles.map(async (fileInfo) => {
          try {
            // If we already have file metadata from the service, use it
            if (
              !fileInfo.isDirectory &&
              fileInfo.size !== undefined &&
              fileInfo.mtime !== undefined
            ) {
              return {
                name: fileInfo.name,
                path: fileInfo.path,
                isDirectory: fileInfo.isDirectory,
                size: fileInfo.size,
                mimeType:
                  fileInfo.mimeType || getMimeType(fileInfo.name) || 'application/octet-stream',
                lastModified: fileInfo.mtime,
                formattedDate: formatDate(fileInfo.mtime),
              };
            }

            // Otherwise get stats for the file
            const stats = await fileSvc.stat(fileInfo.path);
            return {
              name: fileInfo.name,
              path: fileInfo.path,
              isDirectory: fileInfo.isDirectory,
              size: stats.size,
              mimeType: getMimeType(fileInfo.name) || 'application/octet-stream',
              lastModified: stats.mtime,
              formattedDate: formatDate(stats.mtime),
            };
          } catch (statErr) {
            // Don't throw errors for individual files, just report the issue
            fileLogger.warn(`Could not read stats for file: ${fileInfo.path}`, {
              error: (statErr as Error).message,
            });
            return {
              name: fileInfo.name,
              path: fileInfo.path,
              isDirectory: fileInfo.isDirectory,
              error: 'Could not read file info',
              size: 0,
              mimeType: 'unknown',
              lastModified: new Date(0),
              formattedDate: 'unknown',
            };
          }
        }),
      );

      // Apply filtering - only include files, not directories
      let filteredFiles = filesWithMetadata.filter((file) => !file.isDirectory);

      // Filter by file type
      if (filterType) {
        filteredFiles = filteredFiles.filter((file) => file.mimeType.startsWith(filterType));
      }

      // Filter by search term
      if (search) {
        const searchLower = search.toLowerCase();
        filteredFiles = filteredFiles.filter(
          (file) =>
            file.name.toLowerCase().includes(searchLower) ||
            file.mimeType.toLowerCase().includes(searchLower),
        );
      }

      // Filter by date range
      if (dateFrom || dateTo) {
        const fromDate = dateFrom ? new Date(dateFrom) : new Date(0);
        const toDate = dateTo ? new Date(dateTo) : new Date();

        filteredFiles = filteredFiles.filter((file) => {
          if (file.error) return false;
          const fileDate = file.lastModified;
          return fileDate >= fromDate && fileDate <= toDate;
        });
      }

      // Filter by size range
      if (sizeFrom || sizeTo) {
        filteredFiles = filteredFiles.filter((file) => {
          if (file.error) return false;
          if (sizeTo > 0) {
            return file.size >= sizeFrom && file.size <= sizeTo;
          }
          return file.size >= sizeFrom;
        });
      }

      // Apply sorting
      filteredFiles.sort((a, b) => {
        let comparison = 0;

        switch (sort) {
          case 'size':
            comparison = a.size - b.size;
            break;
          case 'date':
            comparison = a.lastModified.getTime() - b.lastModified.getTime();
            break;
          case 'type':
            comparison = a.mimeType.localeCompare(b.mimeType);
            break;
          case 'name':
          default:
            comparison = a.name.localeCompare(b.name);
        }

        return order === 'desc' ? -comparison : comparison;
      });

      // Apply pagination
      const totalFiles = filteredFiles.length;
      const totalPages = Math.ceil(totalFiles / limit);
      const startIndex = (page - 1) * limit;
      const endIndex = Math.min(startIndex + limit, totalFiles);
      const paginatedFiles = filteredFiles.slice(startIndex, endIndex);

      // Map to response format
      const fileDetails = paginatedFiles.map((file) => ({
        name: file.name,
        path: file.path,
        size: file.size,
        mimeType: file.mimeType,
        lastModified: file.formattedDate,
        url: `/api/files/${encodeURIComponent(file.path)}`,
      }));

      const duration = Date.now() - requestStart;

      fileLogger.success('File listing completed', {
        fileCount: {
          total: totalFiles,
          filtered: filteredFiles.length,
          returned: fileDetails.length,
        },
        pagination: {
          page,
          totalPages,
          limit,
        },
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

      // Prepare a more comprehensive response with pagination info
      const response = {
        files: fileDetails,
        pagination: {
          page,
          limit,
          totalFiles,
          totalPages,
          hasNextPage: page < totalPages,
          hasPrevPage: page > 1,
        },
        sorting: {
          sort,
          order,
        },
        filter: {
          type: filterType,
          search,
          dateRange: dateFrom || dateTo ? { from: dateFrom, to: dateTo } : null,
          sizeRange: sizeFrom || sizeTo ? { from: sizeFrom, to: sizeTo } : null,
        },
        _links: {
          self: `/api/files?page=${page}&limit=${limit}&sort=${sort}&order=${order}${filterType ? `&type=${filterType}` : ''}${search ? `&search=${encodeURIComponent(search)}` : ''}`,
          first: `/api/files?page=1&limit=${limit}&sort=${sort}&order=${order}${filterType ? `&type=${filterType}` : ''}${search ? `&search=${encodeURIComponent(search)}` : ''}`,
          last: `/api/files?page=${totalPages}&limit=${limit}&sort=${sort}&order=${order}${filterType ? `&type=${filterType}` : ''}${search ? `&search=${encodeURIComponent(search)}` : ''}`,
          next:
            page < totalPages
              ? `/api/files?page=${page + 1}&limit=${limit}&sort=${sort}&order=${order}${filterType ? `&type=${filterType}` : ''}${search ? `&search=${encodeURIComponent(search)}` : ''}`
              : null,
          prev:
            page > 1
              ? `/api/files?page=${page - 1}&limit=${limit}&sort=${sort}&order=${order}${filterType ? `&type=${filterType}` : ''}${search ? `&search=${encodeURIComponent(search)}` : ''}`
              : null,
        },
      };

      // Use try-catch to handle potential socket errors during response
      try {
        sendResponse(
          sock,
          200,
          {
            'Content-Type': 'application/json',
            'Cache-Control': 'private, max-age=10', // Short cache for listings
            'X-Total-Count': String(totalFiles),
            'X-Total-Pages': String(totalPages),
          },
          JSON.stringify(response),
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
              'Content-Type': 'application/json',
              'Cache-Control': 'no-store',
            },
            JSON.stringify({
              error: 'Failed to list files',
              message: error.message,
              timestamp: formatDate(new Date()),
            }),
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
        storagePath: `${config.staticDir}/${fileName}`,
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
