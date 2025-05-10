// src/modules/file-hosting/fileHostingController.ts
import { Socket } from 'net';
import { Readable } from 'stream';
import * as zlib from 'zlib';
import path from 'path';

import { sendResponse } from '../../entities/sendResponse';
import { IncomingRequest } from '../../entities/http';
import { FileHostingService, FileInfo } from './fileHostingService';
import {
  FileHostingStatsHelper,
  FileStats, // Assuming FileStats is exported
} from './fileHostingStatsHelper';
import { getHeader, getQuery } from '../../utils/httpHelpers';
import { config } from '../../config/server.config';
import { getMimeType } from '../../utils/helpers';
import { formatDate } from '../../utils/dateFormatter';
import {
  evaluateFilter,
  FileFilter, // Assuming FileFilter is exported
} from './file-utils/FileFilter';
import getLoggerInstance from './file-utils/fileHostingLogger';

const log = getLoggerInstance({
  context: 'fileHostingController',
});

// Maximum number of records to fetch from DB if complex JS filtering is needed.
// This is a safeguard against loading the entire DB into memory.
// Adjust this based on typical dataset size and server memory.
const MAX_RECORDS_FOR_JS_FILTERING = 5000;

/* -------------------------------------------------- */
/* BOILER‑PLATE                   */
/* -------------------------------------------------- */
/** Shared helpers that kept re‑appearing everywhere */
const utils = {
  /** Guard: abort handler if the socket is already closed */
  abortIfClosed(sock: Socket, context: Record<string, unknown> = {}): boolean {
    if (sock.destroyed) {
      log.warn('Socket closed before operation', context);
      return true;
    }
    return false;
  },

  /** Wrap sendResponse with try/catch + destroyed check */
  safeSend(
    sock: Socket,
    status: number,
    headers: Record<string, string>,
    body?: string | Buffer | Readable,
  ) {
    if (sock.destroyed) return;
    try {
      sendResponse(sock, status, headers, body);
    } catch (err) {
      log.error('Failed to send response', { error: (err as Error).message });
      if (!sock.destroyed) sock.end();
    }
  },

  /** Common JSON sender */
  json(sock: Socket, status: number, data: unknown, headers: Record<string, string> = {}) {
    this.safeSend(
      sock,
      status,
      { 'Content-Type': 'application/json', ...headers },
      JSON.stringify(data),
    );
  },

  /** Plain‑text sender with a “close” connection header */
  plain(sock: Socket, status: number, message: string) {
    this.safeSend(sock, status, { 'Content-Type': 'text/plain', Connection: 'close' }, message);
  },

  /** Measure duration & auto‑log */
  withTiming<T>(label: string, details: Record<string, unknown>, fn: () => Promise<T>): Promise<T> {
    const start = Date.now();
    return fn().finally(() =>
      log.debug(label, { ...details, duration: `${Date.now() - start}ms` }),
    );
  },

  /** Build “Cache / ETag / Last‑Modified” headers */
  buildCacheHeaders(size: number, mtime: Date) {
    return {
      ETag: `"${size}-${mtime.getTime()}"`,
      'Last-Modified': mtime.toUTCString(),
      'Cache-Control': 'public, max-age=86400, stale-while-revalidate=43200',
    };
  },

  /** One‑liner to decide which compression (if any) to use */
  pickCompression(acceptEncoding: string, mime: string) {
    if (!COMPRESSIBLE_MIME_TYPES.has(mime)) return null;
    if (acceptEncoding.includes('br')) return 'br';
    if (acceptEncoding.includes('gzip')) return 'gzip';
    if (acceptEncoding.includes('deflate')) return 'deflate';
    return null;
  },

  /** Parse pagination / filter / sort query once */
  parseListParams(req: IncomingRequest) {
    const q = req.query;
    return {
      page: Math.max(1, Number(q.page ?? 1)),
      limit: Math.max(1, Number(q.limit ?? 20)),
      sort: (q.sort as string) ?? 'name',
      order: (q.order as string) ?? 'asc',
      filterType: q.type as string, // Simple top-level MIME type filter
      search: q.search as string, // Simple top-level search term
      dateFrom: q.dateFrom as string,
      dateTo: q.dateTo as string,
      sizeFrom: Number(q.sizeFrom ?? 0),
      sizeTo: Number(q.sizeTo ?? 0), // 0 means no upper limit for sizeTo, handle accordingly if needed
      filterOptions: (() => {
        // Complex filter object from 'filter' query param
        try {
          return q.filter ? JSON.parse(q.filter as string) : {};
        } catch {
          log.warn('Invalid filter JSON', { filter: q.filter });
          return {};
        }
      })(),
    };
  },
};

/* -------------------------------------------------- */
/* ONE‑TIME INITIALISATION             */
/* -------------------------------------------------- */

const fileSvc = new FileHostingService(config.staticDir);
const stats = new FileHostingStatsHelper(path.join(process.cwd(), 'data', 'file_stats.db'));

// Export stats for test control and robust async handling
export const __fileHostingStatsHelper = stats;

const statsInitPromise = stats
  .initialize()
  .catch((e) => log.error('Failed to init file‑stats DB', { error: e.message }));
export const __fileHostingStatsHelperInit = statsInitPromise;

const COMPRESSIBLE_MIME_TYPES = new Set([
  'text/plain',
  'text/html',
  'text/css',
  'application/javascript',
  'application/json',
  'application/xml',
  'image/svg+xml',
]);

/* -------------------------------------------------- */
/* CONTROLLER_API                    */
/* -------------------------------------------------- */

export const fileHostingController = {
  /* ------------------------------ HEAD FILE ----------------------------- */
  async headFile(req: IncomingRequest, sock: Socket): Promise<void> {
    if (utils.abortIfClosed(sock, { remote: sock.remoteAddress })) return;
    const fileName = this.resolveFileName(req, sock, { message: 'Missing file param in HEAD' });
    if (!fileName) return;

    await utils
      .withTiming('HEAD processed', { fileName }, async () => {
        const statResult = await fileSvc.stat(fileName);
        const headers: Record<string, string> = {
          'Content-Type': getMimeType(fileName) || 'application/octet-stream',
          'Content-Length': String(statResult.size),
          'Accept-Ranges': 'bytes',
          'X-Content-Type-Options': 'nosniff',
          ...utils.buildCacheHeaders(statResult.size, statResult.mtime),
        };

        const meta = await stats.getStatsByPath(fileName).catch(() => null);
        if (meta?.width) headers['X-Image-Width'] = String(meta.width);
        if (meta?.height) headers['X-Image-Height'] = String(meta.height);
        if (meta?.duration) headers['X-Media-Duration'] = String(meta.duration);

        utils.safeSend(sock, 200, headers);
      })
      .catch((err) => {
        log.error('HEAD failed', { fileName, error: (err as Error).message });
        utils.plain(sock, 404, 'Not found');
      });
  },

  /* ------------------------------ GET FILE ------------------------------ */
  async getFile(req: IncomingRequest, sock: Socket): Promise<void> {
    if (utils.abortIfClosed(sock, { remote: sock.remoteAddress })) return;
    const fileName = this.resolveFileName(req, sock);
    if (!fileName) return;

    const acceptEnc = (getHeader(req, 'accept-encoding') || '').toLowerCase();

    await utils
      .withTiming('GET file', { fileName }, async () => {
        const statResult = await fileSvc.stat(fileName);
        const mime = getMimeType(fileName) || 'application/octet-stream';
        const cacheHeaders = utils.buildCacheHeaders(statResult.size, statResult.mtime);

        const noneMatch = getHeader(req, 'if-none-match');
        const modSince = getHeader(req, 'if-modified-since');
        if (
          noneMatch === cacheHeaders.ETag ||
          (modSince && new Date(modSince) >= statResult.mtime)
        ) {
          utils.safeSend(sock, 304, cacheHeaders);
          return;
        }

        const range = getHeader(req, 'range');
        if (range) return this.sendRange(fileName, statResult, mime, range, sock, acceptEnc);

        const compression = utils.pickCompression(acceptEnc, mime);
        const baseHeaders: Record<string, string> = {
          'Content-Type': mime,
          'Accept-Ranges': 'bytes',
          'X-Content-Type-Options': 'nosniff',
          'Timing-Allow-Origin': '*',
          ...cacheHeaders,
        };

        const source = await fileSvc.readFile(fileName);
        let body: Readable | undefined = source;

        if (compression) {
          const cStream =
            compression === 'br'
              ? zlib.createBrotliCompress()
              : compression === 'gzip'
                ? zlib.createGzip()
                : zlib.createDeflate();
          body = source.pipe(cStream);
          baseHeaders['Content-Encoding'] = compression;
          baseHeaders.Vary = 'Accept-Encoding';
        } else {
          baseHeaders['Content-Length'] = String(statResult.size);
        }
        utils.safeSend(sock, 200, baseHeaders, body);
      })
      .catch((err) => {
        log.error('GET failed', { fileName, error: (err as Error).message });
        utils.plain(sock, 404, 'File not found');
      });
  },

  /* ------------------------- LIST FILES (JSON) -------------------------- */
  async listFiles(req: IncomingRequest, sock: Socket): Promise<void> {
    if (utils.abortIfClosed(sock)) return;

    await utils
      .withTiming('LIST files', { url: req.url }, async () => {
        const p = utils.parseListParams(req);
        log.info('LIST params', { ...p, url: req.url });

        // Determine if a complex filter is present that requires JS-side processing on a larger dataset
        const hasComplexFilter =
          p.filterOptions &&
          Object.keys(p.filterOptions).length > 0 &&
          (p.filterOptions.and ||
            p.filterOptions.or ||
            p.filterOptions.not ||
            p.filterOptions.regex);

        const queryOptsForDB: Record<string, unknown> = {
          // sortBy and sortOrder are for the DB query if it does the final sort
          // If JS sorts later, DB sort is just for a consistent large batch
          sortBy: p.sort,
          sortOrder: p.order,
        };

        // If complex JS filtering will be done, fetch a larger dataset without DB pagination
        // but still try to pre-filter with simple conditions at DB level if possible.
        if (hasComplexFilter) {
          log.debug('Complex filter detected. Fetching larger dataset for JS filtering.', {
            filterOptions: p.filterOptions,
          });
          queryOptsForDB.limit = MAX_RECORDS_FOR_JS_FILTERING;
          queryOptsForDB.offset = 0; // Start from the beginning for JS filtering
          // Pass simple top-level filters to queryFileStats if they exist
          if (p.filterType) queryOptsForDB.mimeType = p.filterType;
          if (p.sizeFrom) queryOptsForDB.minSize = p.sizeFrom;
          if (p.sizeTo) queryOptsForDB.maxSize = p.sizeTo;
          // Note: queryFileStats currently does not use p.filterOptions directly for its SQL
          // It only uses the explicit simple keys like mimeType, minSize.
          // We are now relying on fetching a large batch and then using evaluateFilter in JS.
        } else {
          // No complex filter, or simple filter that queryFileStats can handle.
          // Fetch a limited set for the current page (e.g., p.limit * 2 as before).
          // The `evaluateFilter` will still run, but on a smaller, pre-filtered (by DB) set.
          queryOptsForDB.limit = p.limit * 2; // Fetch a bit more for potential filtering
          queryOptsForDB.offset = (p.page - 1) * p.limit;
          if (p.filterType) queryOptsForDB.mimeType = p.filterType;
          // Spread p.filterOptions if it contains simple key-value pairs queryFileStats might understand
          // (e.g., if filter={"mimeType":"image/"} was passed instead of type=image/)
          // However, the main `filterOptions` are for `evaluateFilter`.
          if (p.filterOptions && Object.keys(p.filterOptions).length > 0 && !hasComplexFilter) {
            // If filterOptions are simple (e.g. {mimeType: 'image/', minSize: 100}), spread them
            // This relies on queryFileStats being able to pick up these simple top-level keys
            Object.assign(queryOptsForDB, p.filterOptions);
          }
          if (p.sizeFrom) queryOptsForDB.minSize = p.sizeFrom;
          if (p.sizeTo) queryOptsForDB.maxSize = p.sizeTo;
        }

        let allPotentiallyMatchingFiles: FileStats[];
        try {
          allPotentiallyMatchingFiles = await stats.queryFileStats(queryOptsForDB);
          log.debug('LIST raw data from DB', {
            count: allPotentiallyMatchingFiles.length,
            queryOptsForDB,
          });
        } catch (dbError) {
          log.error('Failed to query file stats from DB', {
            error: (dbError as Error).message,
            queryOptsForDB,
          });
          allPotentiallyMatchingFiles = []; // Proceed with empty if DB fails
        }

        if (allPotentiallyMatchingFiles.length === 0 && !hasComplexFilter && !p.filterType) {
          // Fallback to filesystem list if DB is empty AND no significant filters were applied that might explain empty results
          log.info(
            'DB returned no results and no major filters applied, attempting filesystem fallback.',
          );
          const fsList = await fileSvc.listFiles('.', false); // Non-recursive for root, adjust if needed
          allPotentiallyMatchingFiles = (fsList as FileInfo[])
            .filter((f) => !f.isDirectory)
            .map((f) => ({
              // Map to FileStats structure
              fileName: f.name,
              filePath: f.path,
              size: f.size ?? 0,
              mimeType: getMimeType(f.name) || 'application/octet-stream',
              lastModified: f.mtime ?? new Date(),
              createdAt: f.mtime ?? new Date(), // Approx
              updatedAt: f.mtime ?? new Date(), // Approx
              // width, height, duration etc. would be undefined from basic FS listing
            }));
          log.debug('Filesystem fallback data', { count: allPotentiallyMatchingFiles.length });
        }

        log.info('RAW FILES passed to JS filtering', {
          count: allPotentiallyMatchingFiles.length,
          firstFew: allPotentiallyMatchingFiles
            .slice(0, 3)
            .map((f) => ({ fN: f.fileName, mT: f.mimeType })),
        });

        let jsFilteredResults = allPotentiallyMatchingFiles;

        // Apply simple text search (if any)
        const searchTermLower = p.search?.toLowerCase();
        if (searchTermLower) {
          jsFilteredResults = jsFilteredResults.filter(
            (f) =>
              f.fileName.toLowerCase().includes(searchTermLower) ||
              f.mimeType.toLowerCase().includes(searchTermLower),
          );
        }

        // Apply date range filtering (if any)
        if (p.dateFrom || p.dateTo) {
          const from = p.dateFrom ? new Date(p.dateFrom) : new Date(0); // начало времен
          const to = p.dateTo ? new Date(p.dateTo) : new Date(Date.now() + 8640000000); // Далекое будущее
          jsFilteredResults = jsFilteredResults.filter((f) => {
            const lm = f.lastModified instanceof Date ? f.lastModified : new Date(f.lastModified);
            return lm >= from && lm <= to;
          });
        }

        // Apply complex filter options using evaluateFilter
        if (p.filterOptions && Object.keys(p.filterOptions).length > 0) {
          const initialCount = jsFilteredResults.length;
          jsFilteredResults = jsFilteredResults.filter((f) =>
            evaluateFilter(f, p.filterOptions as FileFilter),
          );
          log.debug('JS filtering applied', {
            initialCount,
            finalCount: jsFilteredResults.length,
            filter: p.filterOptions,
          });
        }
        log.info('FILTERED FILES after JS evaluateFilter', {
          count: jsFilteredResults.length,
          firstFew: jsFilteredResults.slice(0, 3).map((f) => ({ fN: f.fileName, mT: f.mimeType })),
        });
        // Define allowed sort keys and their types for FileStats
        type SortableFileStatsKey = keyof Pick<
          FileStats,
          'fileName' | 'filePath' | 'mimeType' | 'size' | 'lastModified' | 'createdAt' | 'updatedAt'
        >;

        // Ensure the sort key is valid, fallback to 'fileName' if not
        const sortKey: SortableFileStatsKey = (
          [
            'fileName',
            'filePath',
            'mimeType',
            'size',
            'lastModified',
            'createdAt',
            'updatedAt',
          ].includes(p.sort)
            ? p.sort
            : 'fileName'
        ) as SortableFileStatsKey;

        // Sort the JS-filtered results using type-safe access
        jsFilteredResults.sort((a, b) => {
          const valA = a[sortKey];
          const valB = b[sortKey];

          let comparison = 0;
          if (typeof valA === 'string' && typeof valB === 'string') {
            comparison = valA.localeCompare(valB);
          } else if (valA instanceof Date && valB instanceof Date) {
            comparison = valA.getTime() - valB.getTime();
          } else if (typeof valA === 'number' && typeof valB === 'number') {
            comparison = valA - valB;
          } else {
            // Fallback for undefined or mismatched types
            comparison = String(valA ?? '').localeCompare(String(valB ?? ''));
          }
          return p.order === 'desc' ? -comparison : comparison;
        });

        // Paginate the JS-filtered and sorted results
        const totalFilteredFiles = jsFilteredResults.length;
        const totalPages = Math.ceil(totalFilteredFiles / p.limit);
        const startIndex = (p.page - 1) * p.limit;
        const endIndex = startIndex + p.limit;
        const pageDataArray = jsFilteredResults.slice(startIndex, endIndex);

        const pageData = pageDataArray.map((f) => ({
          name: f.fileName,
          path: f.filePath,
          size: f.size,
          mimeType: f.mimeType,
          lastModified: formatDate(f.lastModified),
          url: `/api/files/${encodeURIComponent(f.filePath)}`, // Ensure filePath is valid for URL
        }));

        const hasNextPage = p.page < totalPages;
        const urlObj = new URL(req.url); // Use full req.url
        urlObj.searchParams.set('page', String(p.page + 1));
        // Preserve other query params for next/prev links
        // No need to set limit, sort, order again if they are already in urlObj.searchParams

        const nextLink = hasNextPage ? urlObj.toString() : null; // Use full URL for link
        urlObj.searchParams.set('page', String(p.page - 1));
        const prevLink = p.page > 1 ? urlObj.toString() : null;

        utils.json(sock, 200, {
          files: pageData,
          pagination: {
            page: p.page,
            limit: p.limit,
            totalFiles: totalFilteredFiles,
            totalPages,
            hasNextPage,
            hasPrevPage: p.page > 1,
          },
          _links: {
            self: req.url, // Current request URL
            next: nextLink,
            prev: prevLink,
          },
        });
      })
      .catch((err) => {
        log.error('LIST files failed', {
          error: (err as Error).message,
          stack: (err as Error).stack,
        });
        utils.plain(sock, 500, 'Failed to list files');
      });
  },

  /* ----------------------------- UPLOAD -------------------------------- */
  async uploadFile(req: IncomingRequest, sock: Socket): Promise<void> {
    if (utils.abortIfClosed(sock)) return;
    const fileNameHeader = getHeader(req, 'x-filename'); // Case-insensitive getHeader
    const fileNameQuery = getQuery(req, 'file');
    const fileName = fileNameQuery || fileNameHeader;

    if (!fileName) {
      utils.plain(sock, 400, 'Missing file name (expected in ?file= query or X-Filename header)');
      return;
    }

    await utils
      .withTiming('UPLOAD', { fileName }, async () => {
        const mime = getMimeType(fileName) || '';
        // Allow any file type for now, or add specific checks
        // if (!/^image\/|^video\/|^audio\//.test(mime)) {
        //   utils.plain(sock, 400, 'Only media files allowed');
        //   return;
        // }
        if (!req.body || !(req.body instanceof Buffer) || req.body.length === 0) {
          utils.plain(sock, 400, 'Empty or invalid body');
          return;
        }

        async function* toChunks(buf: Buffer) {
          const CHUNK_SIZE = 1 << 20; // 1MB
          for (let i = 0; i < buf.length; i += CHUNK_SIZE) {
            yield buf.subarray(i, i + CHUNK_SIZE);
          }
        }

        await fileSvc.saveFile(fileName, toChunks(req.body));

        // Asynchronously collect and persist stats, don't block response
        const absolutePath = path.join(config.staticDir, fileName);
        stats
          .getFileStats(absolutePath, config.staticDir)
          .then((meta) => stats.saveFileStats(meta))
          .then((fileId) => log.info('Stats saved post-upload', { fileName, fileId }))
          .catch((statErr) =>
            log.error('Failed to save stats post-upload', {
              fileName,
              error: (statErr as Error).message,
            }),
          );

        utils.json(sock, 201, {
          // 201 Created for successful upload
          success: true,
          fileName,
          size: req.body.length,
          mimeType: mime, // Use detected mime
          message: 'File uploaded successfully.',
        });
      })
      .catch((err) => {
        log.error('UPLOAD failed', {
          fileName,
          error: (err as Error).message,
          stack: (err as Error).stack,
        });
        utils.plain(sock, 500, `Upload failed: ${(err as Error).message}`);
      });
  },

  /* ----------------------------- DELETE -------------------------------- */
  async deleteFile(req: IncomingRequest, sock: Socket): Promise<void> {
    if (utils.abortIfClosed(sock)) return;
    const fileName = this.resolveFileName(req, sock); // Use resolveFileName for consistency
    if (!fileName) {
      // resolveFileName already sends a 400 response
      return;
    }

    await utils
      .withTiming('DELETE', { fileName }, async () => {
        await fileSvc.deleteFile(fileName); // This will throw if file not found / deletion fails

        // Asynchronously delete stats, don't block response on this
        stats
          .deleteFileStats(fileName)
          .then((deleted) => log.info('Stats deleted post-file-delete', { fileName, deleted }))
          .catch((statErr) =>
            log.error('Failed to delete stats post-file-delete', {
              fileName,
              error: (statErr as Error).message,
            }),
          );

        utils.json(sock, 200, {
          success: true,
          fileName,
          message: 'File deleted successfully',
        });
      })
      .catch((err) => {
        log.error('DELETE failed', { fileName, error: (err as Error).message });
        // Check if the error message indicates file not found for a 404
        if (
          (err as Error).message.toLowerCase().includes('not found') ||
          (err as Error).message.includes('ENOENT')
        ) {
          utils.plain(sock, 404, 'File not found');
        } else {
          utils.plain(sock, 500, `Deletion failed: ${(err as Error).message}`);
        }
      });
  },

  /* ------------------------- RESOLVE_FILE_NAME ------------------------- */
  resolveFileName(
    req: IncomingRequest,
    sock: Socket, // Keep sock for sending error response
    extra?: { message: string },
  ): string | undefined {
    // Prioritize path parameter (e.g., /api/files/:filename)
    let name = (req.ctx?.params as Record<string, string>)?.filename;

    // Fallback to query parameter (e.g., /api/files?file=filename)
    if (!name) {
      const fileQuery = getQuery(req, 'file');
      if (fileQuery) {
        name = decodeURIComponent(fileQuery);
      }
    }

    if (name) {
      // Basic path sanitization: prevent directory traversal
      // Resolve turns '..' into actual paths, then we check if it's still within a safe root.
      // For this controller, names are typically relative to staticDir.
      const safeName = path.normalize(name).replace(/^(\.\.[/\\])+/, '');
      if (name !== safeName) {
        log.warn('Potential path traversal attempt blocked', {
          original: name,
          sanitized: safeName,
          url: req.url,
        });
        utils.plain(sock, 400, 'Invalid file path.');
        return undefined;
      }
      return safeName;
    }

    log.warn('File name missing', { url: req.url, method: req.method, ...(extra || {}) });
    utils.plain(
      sock,
      400,
      extra?.message || 'File name is required in path or as "file" query parameter.',
    );
    return undefined;
  },

  /* ------------------------- RANGE_SENDER_HELPER ----------------------- */
  async sendRange(
    fileName: string,
    statResult: Awaited<ReturnType<typeof fileSvc.stat>>, // Correct type
    mime: string,
    rangeHdr: string,
    sock: Socket,
    acceptEnc: string, // Keep acceptEnc if needed for other parts, though not needed here
  ) {
    const size = statResult.size;
    const rangeMatch = /bytes=(\d*)-(\d*)/.exec(rangeHdr);
    if (!rangeMatch) {
      utils.plain(sock, 400, 'Malformed Range header');
      return;
    }

    const [, startStr, endStr] = rangeMatch;
    let start = startStr ? parseInt(startStr, 10) : 0;
    let end = endStr ? parseInt(endStr, 10) : size - 1;

    if (isNaN(start) || isNaN(end) || start > end || start < 0 || end >= size) {
      utils.plain(sock, 416, 'Range Not Satisfiable'); // HTTP 416
      return;
    }

    // Adjust for "bytes=-N" (last N bytes) or "bytes=N-" (from N to end)
    if (startStr === '' && endStr !== '') {
      // bytes=-N
      start = Math.max(0, size - parseInt(endStr, 10));
      end = size - 1;
    } else if (startStr !== '' && endStr === '') {
      // bytes=N-
      end = size - 1;
    }

    const contentLength = end - start + 1;
    const source = await fileSvc.readFile(fileName, { start, end });

    const headers = {
      'Content-Type': mime,
      'Content-Length': String(contentLength),
      'Content-Range': `bytes ${start}-${end}/${size}`,
      'Accept-Ranges': 'bytes',
      'Accept-Encoding': acceptEnc, // Not typically needed for range requests, compression handled differently
      ...utils.buildCacheHeaders(size, statResult.mtime), // Cache headers still useful
    };

    utils.safeSend(sock, 206, headers, source); // HTTP 206 Partial Content
  },
};

/* eof */
