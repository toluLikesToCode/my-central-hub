// src/modules/file-hosting/fileHostingController.ts
import { Socket } from 'net';
import { sendResponse } from '../../entities/sendResponse';
import { IncomingRequest } from '../../entities/http';
import { FileHostingService, FileInfo } from './fileHostingService';
import { FileHostingStatsHelper } from './fileHostingStatsHelper';
import { getHeader, getQuery } from '../../utils/httpHelpers';
import { config } from '../../config/server.config';
import logger from '../../utils/logger';
import { getMimeType } from '../../utils/helpers';
import { Readable } from 'stream';
import { formatDate } from '../../utils/dateFormatter';
// Import zlib for compression
import * as zlib from 'zlib';
import path from 'path';

// Create module-specific logger with metadata
const fileLogger = logger.child({
  module: 'file-hosting',
  component: 'controller',
  feature: 'media-files',
});

// Use staticDir instead of mediaDir for file hosting
const fileSvc = new FileHostingService(config.staticDir);

// Initialize the stats helper with the database path
const statsHelper = new FileHostingStatsHelper(path.join(process.cwd(), 'data', 'file_stats.db'));

// Initialize the database when the controller is loaded
statsHelper.initialize().catch((error) => {
  fileLogger.error('Failed to initialize file stats database', {
    error: error.message,
    stack: error.stack,
  });
});

// Define compressible MIME types
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
   * Handles HEAD requests for files - returns same headers as GET but no body
   * This allows clients to efficiently get metadata without downloading content
   *
   * @param req - The incoming HTTP request with file query parameter
   * @param sock - The TCP socket to write the response to
   * @returns {Promise<void>} Resolves when headers are sent
   */
  async headFile(req: IncomingRequest, sock: Socket): Promise<void> {
    const requestStart = Date.now();
    const fileName = this.resolveFileName(req, sock, {
      message: 'Missing file parameter in HEAD request',
    });
    if (!fileName) return;

    try {
      // Get file stats but don't send the actual file
      const fileStat = await fileSvc.stat(fileName);
      const mimeType = getMimeType(fileName) || 'application/octet-stream';
      const lastModified = fileStat.mtime.toUTCString();
      const etag = `"${fileStat.size}-${fileStat.mtime.getTime()}"`;

      const headers: Record<string, string> = {
        'Content-Type': mimeType,
        'Content-Length': String(fileStat.size),
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'public, max-age=86400, stale-while-revalidate=43200',
        ETag: etag,
        'Last-Modified': lastModified,
        'X-Content-Type-Options': 'nosniff',
      };

      // For media files, add extra headers with basic dimensions/duration if available
      try {
        const fileStats = await statsHelper.getStatsByPath(fileName);
        if (fileStats) {
          if (fileStats.width && fileStats.height) {
            headers['X-Image-Width'] = String(fileStats.width);
            headers['X-Image-Height'] = String(fileStats.height);
          }
          if (fileStats.duration) {
            headers['X-Media-Duration'] = String(fileStats.duration);
          }
        }
      } catch (statsErr) {
        // Continue even if stats lookup fails
        fileLogger.debug(`Stats lookup failed during HEAD request: ${fileName}`, {
          error: (statsErr as Error).message,
        });
      }

      // Send response with headers but no body
      sendResponse(sock, 200, headers, undefined);

      const duration = Date.now() - requestStart;
      fileLogger.debug(`HEAD request processed for "${fileName}"`, {
        fileName,
        duration: `${duration}ms`,
      });
    } catch (err) {
      fileLogger.error(`HEAD request failed for "${fileName}"`, {
        error: (err as Error).message,
      });

      if (!sock.destroyed) {
        sendResponse(
          sock,
          404,
          { 'Content-Type': 'text/plain' },
          undefined, // No body for HEAD
        );
      }
    }
  },

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

    // first check if the IncomingRequest has the property ctx.params.filename
    const fileName: string | undefined = this.resolveFileName(req, sock);
    if (!fileName) return;

    // Extract headers
    const ifNoneMatchHeader = getHeader(req, 'if-none-match');
    const ifModifiedSinceHeader = getHeader(req, 'if-modified-since');
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

      const responseHeaders: Record<string, string> = {
        'Content-Type': mimeType,
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'public, max-age=86400, stale-while-revalidate=43200',
        ETag: etag,
        'Last-Modified': lastModified,
        'X-Content-Type-Options': 'nosniff',
        'Timing-Allow-Origin': '*',
      };

      // Optimize for specific content types with aggressive caching
      if (mimeType.startsWith('video/')) {
        // Video files: long cache time, optimized for streaming
        responseHeaders['Cache-Control'] = 'public, max-age=604800, immutable';
        responseHeaders['X-Accel-Buffering'] = 'yes';
        responseHeaders['Accept-Ranges'] = 'bytes';
      } else if (mimeType.startsWith('image/gif')) {
        // GIFs: long cache time, preload hint
        responseHeaders['Cache-Control'] = 'public, max-age=31536000, immutable';
        responseHeaders['X-Content-Type-Options'] = 'nosniff';
        responseHeaders['Timing-Allow-Origin'] = '*';
      } else if (mimeType.startsWith('image/')) {
        // Images: longest cache time, no revalidation needed for most cases
        responseHeaders['Cache-Control'] = 'public, max-age=31536000, immutable';
        responseHeaders['X-Content-Type-Options'] = 'nosniff';
      }

      // Add preconnect hint for common CDN domains if using a CDN
      responseHeaders['Link'] = '<https://your-cdn-domain.com>; rel=preconnect';

      // Early hints for browser optimization
      if (mimeType.startsWith('video/')) {
        responseHeaders['X-Media-Buffer-Hint'] = '10';
      }

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
          const suffix = parseInt(endStr, 10);
          if (isNaN(suffix) || suffix <= 0) {
            fileLogger.warn('Invalid suffix range header format', {
              rangeHeader: rangeHdr,
              fileName,
            });
            sendResponse(
              sock,
              416,
              { 'Content-Range': `bytes */${size}` },
              '416 Range Not Satisfiable',
            );
            return;
          }
          start = Math.max(0, size - suffix);
          end = size - 1;
        }

        // Video chunk optimization: adjust start/end for large video ranges
        if (mimeType.startsWith('video/')) {
          const chunkSize = 1024 * 1024; // 1MB chunks for video
          if (end - start > chunkSize * 2) {
            end = start + Math.ceil((end - start) / chunkSize) * chunkSize;
            end = Math.min(end, size - 1);
          }
        }

        let stream: Readable;
        try {
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

        responseHeaders['Content-Range'] = `bytes ${start}-${end}/${size}`;
        responseHeaders['Content-Length'] = String(len);

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

        return;
      }

      let responseStream: Readable;
      let sourceStream: Readable;

      if (isBinaryContent && mimeType.startsWith('image/')) {
        responseHeaders['Cache-Control'] = 'public, max-age=31536000, immutable';
      }

      let compressionEncoding: string | null = null;
      if (isCompressible(mimeType)) {
        if (supportsBrotli) compressionEncoding = 'br';
        else if (supportsGzip) compressionEncoding = 'gzip';
        else if (supportsDeflate) compressionEncoding = 'deflate';
      }

      try {
        sourceStream = await fileSvc.readFile(fileName);
        if (!sourceStream) throw new Error('Source stream is undefined');

        if (compressionEncoding) {
          let compressStream: zlib.BrotliCompress | zlib.Gzip | zlib.Deflate;
          if (compressionEncoding === 'br') {
            compressStream = zlib.createBrotliCompress();
          } else if (compressionEncoding === 'gzip') {
            compressStream = zlib.createGzip();
          } else {
            compressStream = zlib.createDeflate();
          }

          responseStream = sourceStream.pipe(compressStream);

          responseHeaders['Content-Encoding'] = compressionEncoding;
          responseHeaders['Vary'] = 'Accept-Encoding';

          fileLogger.debug(`Serving compressed content (${compressionEncoding})`, {
            fileName,
            mimeType,
            originalSize: size,
          });

          compressStream.on('error', (err: Error) => {
            fileLogger.error(
              `Compression stream error for "${fileName}" (${compressionEncoding})`,
              {
                error: err.message,
                stack: err.stack,
              },
            );
            if (!sourceStream.destroyed) sourceStream.destroy();
            if (!sock.destroyed) sock.destroy(err);
          });
        } else {
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

      sourceStream.on('error', (err: Error) => {
        fileLogger.error(`Source stream error for "${fileName}"`, {
          error: err.message,
          stack: err.stack,
        });
        if (responseStream !== sourceStream && !responseStream.destroyed) {
          responseStream.destroy(err);
        }
        if (!sock.destroyed) {
          sock.destroy(err);
        }
      });

      if (sock.destroyed) {
        fileLogger.warn('Socket closed before sending full content', { fileName });
        if (responseStream && !responseStream.destroyed) responseStream.destroy();
        if (sourceStream && responseStream !== sourceStream && !sourceStream.destroyed)
          sourceStream.destroy();
        return;
      }

      if (isBinaryContent && size > 1024 * 1024) {
        responseHeaders['Connection'] = 'close';
      }

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
        if (responseStream && !responseStream.destroyed) {
          responseStream.destroy();
          fileLogger.warn(`Client disconnected during full transfer: "${fileName}"`, {
            duration: `${Date.now() - requestStart}ms`,
            encoding: compressionEncoding || 'none',
          });
        }
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
      if (sock.destroyed) {
        fileLogger.warn('Socket already closed, aborting file listing operation', {
          remoteAddress: sock.remoteAddress,
        });
        return;
      }

      const allFilesResult = await fileSvc.listFiles();

      // Handle both array of strings (from tests) and array of FileInfo objects
      let allFiles: FileInfo[] = [];

      if (Array.isArray(allFilesResult)) {
        if (allFilesResult.length > 0 && typeof allFilesResult[0] === 'string') {
          // Convert string[] to FileInfo[]
          allFiles = (allFilesResult as string[]).map((filename) => ({
            name: filename,
            path: filename,
            isDirectory: false,
          }));
        } else {
          // It's already FileInfo[]
          allFiles = allFilesResult as FileInfo[];
        }
      }

      const filesWithMetadata = await Promise.all(
        allFiles.map(async (fileInfo) => {
          try {
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

      let filteredFiles = filesWithMetadata.filter((file) => !file.isDirectory);

      if (filterType) {
        filteredFiles = filteredFiles.filter((file) => file.mimeType.startsWith(filterType));
      }

      if (search) {
        const searchLower = search.toLowerCase();
        filteredFiles = filteredFiles.filter(
          (file) =>
            file.name.toLowerCase().includes(searchLower) ||
            file.mimeType.toLowerCase().includes(searchLower),
        );
      }

      if (dateFrom || dateTo) {
        const fromDate = dateFrom ? new Date(dateFrom) : new Date(0);
        const toDate = dateTo ? new Date(dateTo) : new Date();

        filteredFiles = filteredFiles.filter((file) => {
          if (file.error) return false;
          const fileDate = file.lastModified;
          return fileDate >= fromDate && fileDate <= toDate;
        });
      }

      if (sizeFrom || sizeTo) {
        filteredFiles = filteredFiles.filter((file) => {
          if (file.error) return false;
          if (sizeTo > 0) {
            return file.size >= sizeFrom && file.size <= sizeTo;
          }
          return file.size >= sizeFrom;
        });
      }

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

      const totalFiles = filteredFiles.length;
      const totalPages = Math.ceil(totalFiles / limit);
      const startIndex = (page - 1) * limit;
      const endIndex = Math.min(startIndex + limit, totalFiles);
      const paginatedFiles = filteredFiles.slice(startIndex, endIndex);

      const fileDetails = paginatedFiles.map((file) => ({
        name: file.name,
        path: file.path,
        size: file.size,
        mimeType: file.mimeType,
        lastModified: file.formattedDate,
        url: `/api/files/${encodeURIComponent(file.path.toString())}`,
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

      if (sock.destroyed) {
        fileLogger.warn('Socket disconnected during file listing operation', {
          remoteAddress: sock.remoteAddress,
          duration: `${duration}ms`,
        });
        return;
      }

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

      try {
        sendResponse(
          sock,
          200,
          {
            'Content-Type': 'application/json',
            'Cache-Control': 'private, max-age=10',
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

      // Collect and store file stats after successful upload
      try {
        const absolutePath = path.join(config.staticDir, fileName);
        const fileStats = await statsHelper.getFileStats(absolutePath, config.staticDir);
        await statsHelper.saveFileStats(fileStats);

        fileLogger.debug('File statistics collected and stored', {
          fileName,
          mimeType: fileStats.mimeType,
          size: fileStats.size,
          width: fileStats.width,
          height: fileStats.height,
          duration: fileStats.duration,
        });
      } catch (statsErr) {
        // Log but don't fail the upload if stats collection fails
        fileLogger.warn('Failed to collect file statistics', {
          fileName,
          error: (statsErr as Error).message,
        });
      }

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

      if (sock.destroyed) {
        fileLogger.warn('Socket closed before file deletion', {
          fileName,
          remoteAddress: sock.remoteAddress,
        });
        return;
      }

      // Delete the file
      await fileSvc.deleteFile(fileName);

      // Also remove file statistics from database
      try {
        const wasDeleted = await statsHelper.deleteFileStats(fileName);
        fileLogger.debug(
          `File statistics ${wasDeleted ? 'deleted' : 'not found'} for file: ${fileName}`,
        );
      } catch (statsErr) {
        // Just log error but don't stop the operation
        fileLogger.warn('Failed to delete file statistics', {
          fileName,
          error: (statsErr as Error).message,
        });
      }

      const duration = Date.now() - requestStart;
      fileLogger.success(`File "${fileName}" deleted successfully`, {
        ...fileInfo,
        duration: `${duration}ms`,
      });

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

  /**
   * Resolves the file name from an incoming HTTP request.
   *
   * This function attempts to extract a file name from the request in the following priority order:
   *   1. `req.ctx.params.filename` (if available)
   *   2. The query parameter `file` (if present, it will be URL-decoded)
   *
   * If neither source provides a valid file name, it logs a warning and sends a `400 Bad Request` response to the client,
   * indicating that the file name is required.
   *
   * @param req - The IncomingRequest object that contains request context, path, URL, and query parameters.
   *              Expected to potentially have `ctx.params.filename` or a query parameter named `file`.
   * @param sock - The socket over which the response can be sent back to the client in case of an error.
   * @param p0 - Optional parameter for additional logging context, such as a message.
   *
   * @returns The resolved file name as a decoded string if found, otherwise `undefined`. If `undefined`, a response has already been sent back to the client.
   *
   * @example
   * const fileName = resolveFileName(req, sock);
   * if (!fileName) return; // response has already been handled
   */
  resolveFileName(req: IncomingRequest, sock: Socket, p0?: { message: string }) {
    return (
      (req.ctx?.params as Record<string, string>)?.filename ??
      (getQuery(req, 'file') ? decodeURIComponent(getQuery(req, 'file')!) : undefined) ??
      (() => {
        fileLogger.warn('File name is required but not provided', {
          url: req.url,
          path: req.path,
          ...(p0?.message ? { message: p0.message } : {}),
        });
        sendResponse(
          sock,
          400,
          { 'Content-Type': 'text/plain', Connection: 'close' },
          'File name is required.',
        );
        return undefined;
      })()
    );
  },
};
