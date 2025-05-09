// src/modules/file-hosting/fileHostingController.ts
import { Socket } from 'net';
import { Readable } from 'stream';
import * as zlib from 'zlib';
import path from 'path';

import { sendResponse } from '../../entities/sendResponse';
import { IncomingRequest } from '../../entities/http';
import { FileHostingService, FileInfo } from './fileHostingService';
import { FileHostingStatsHelper } from './fileHostingStatsHelper';
import { getHeader, getQuery } from '../../utils/httpHelpers';
import { config } from '../../config/server.config';
import logger from '../../utils/logger';
import { getMimeType } from '../../utils/helpers';
import { formatDate } from '../../utils/dateFormatter';
import { evaluateFilter } from './file-utils/FileFilter';

/* -------------------------------------------------- */
/*                     BOILER‑PLATE                   */
/* -------------------------------------------------- */

const log = logger.child({
  module: 'file-hosting',
  component: 'controller',
  feature: 'media-files',
});

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
      page: Number(q.page ?? 1),
      limit: Number(q.limit ?? 20),
      sort: (q.sort as string) ?? 'name',
      order: (q.order as string) ?? 'asc',
      filterType: q.type as string,
      search: q.search as string,
      dateFrom: q.dateFrom as string,
      dateTo: q.dateTo as string,
      sizeFrom: Number(q.sizeFrom ?? 0),
      sizeTo: Number(q.sizeTo ?? 0),
      filterOptions: (() => {
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
/*                ONE‑TIME INITIALISATION             */
/* -------------------------------------------------- */

const fileSvc = new FileHostingService(config.staticDir);
const stats = new FileHostingStatsHelper(path.join(process.cwd(), 'data', 'file_stats.db'));
stats.initialize().catch((e) => log.error('Failed to init file‑stats DB', { error: e.message }));

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
/*                  CONTROLLER_API                    */
/* -------------------------------------------------- */

export const fileHostingController = {
  /* ------------------------------ HEAD FILE ----------------------------- */
  async headFile(req: IncomingRequest, sock: Socket): Promise<void> {
    if (utils.abortIfClosed(sock, { remote: sock.remoteAddress })) return;
    const fileName = this.resolveFileName(req, sock, { message: 'Missing file param in HEAD' });
    if (!fileName) return;

    await utils
      .withTiming('HEAD processed', { fileName }, async () => {
        const stat = await fileSvc.stat(fileName);
        const headers: Record<string, string> = {
          'Content-Type': getMimeType(fileName) || 'application/octet-stream',
          'Content-Length': String(stat.size),
          'Accept-Ranges': 'bytes',
          'X-Content-Type-Options': 'nosniff',
          ...utils.buildCacheHeaders(stat.size, stat.mtime),
        };

        // Optional extra x‑ headers
        const meta = await stats.getStatsByPath(fileName).catch(() => null);
        if (meta?.width) headers['X-Image-Width'] = String(meta.width);
        if (meta?.height) headers['X-Image-Height'] = String(meta.height);
        if (meta?.duration) headers['X-Media-Duration'] = String(meta.duration);

        utils.safeSend(sock, 200, headers);
      })
      .catch((err) => {
        log.error('HEAD failed', { fileName, error: err.message });
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
        const stat = await fileSvc.stat(fileName);
        const mime = getMimeType(fileName) || 'application/octet-stream';
        const cacheHeaders = utils.buildCacheHeaders(stat.size, stat.mtime);

        /* --- 304 short‑circuit --- */
        const noneMatch = getHeader(req, 'if-none-match');
        const modSince = getHeader(req, 'if-modified-since');
        if (noneMatch === cacheHeaders.ETag || (modSince && new Date(modSince) >= stat.mtime)) {
          utils.safeSend(sock, 304, cacheHeaders);
          return;
        }

        /* --- Range handling simplified to delegate to helper below --- */
        const range = getHeader(req, 'range');
        if (range) return this.sendRange(fileName, stat, mime, range, sock, acceptEnc);

        /* --- Full file path --- */
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
          baseHeaders['Content-Length'] = String(stat.size);
        }

        utils.safeSend(sock, 200, baseHeaders, body);
      })
      .catch((err) => {
        log.error('GET failed', { fileName, error: err.message });
        utils.plain(sock, 404, 'File not found');
      });
  },

  /* ------------------------- LIST FILES (JSON) -------------------------- */
  async listFiles(req: IncomingRequest, sock: Socket): Promise<void> {
    if (utils.abortIfClosed(sock)) return;

    await utils
      .withTiming('LIST files', { url: req.url }, async () => {
        const p = utils.parseListParams(req);

        // combine DB query + optional filesystem fallback
        const queryOpts: Record<string, unknown> = {
          limit: p.limit * 2,
          offset: (p.page - 1) * p.limit,
          ...(p.filterOptions || {}),
        };
        if (p.filterType) queryOpts.mimeType ??= p.filterType;
        if (p.sizeFrom) queryOpts.minSize ??= p.sizeFrom;
        if (p.sizeTo) queryOpts.maxSize ??= p.sizeTo;

        let results = await stats.queryFileStats(queryOpts);

        // extra filtering / sorting in JS (search, date range, etc.)
        const searchLower = p.search?.toLowerCase();
        if (searchLower)
          results = results.filter(
            (f) =>
              f.fileName.toLowerCase().includes(searchLower) ||
              f.mimeType.toLowerCase().includes(searchLower),
          );

        if (p.dateFrom || p.dateTo) {
          const from = p.dateFrom ? new Date(p.dateFrom) : new Date(0);
          const to = p.dateTo ? new Date(p.dateTo) : new Date();
          results = results.filter((f) => f.lastModified >= from && f.lastModified <= to);
        }
        // advanced filter options via JSON filter param
        if (p.filterOptions && Object.keys(p.filterOptions).length) {
          results = results.filter((f) => evaluateFilter(f, p.filterOptions));
        }

        results.sort((a, b) => {
          const cmp = (key: keyof typeof a) =>
            key === 'fileName' || key === 'mimeType'
              ? (a[key] as unknown as string).localeCompare(b[key] as unknown as string)
              : (a[key] as number) - (b[key] as number);

          const key =
            p.sort === 'size'
              ? 'size'
              : p.sort === 'date'
                ? 'lastModified'
                : p.sort === 'type'
                  ? 'mimeType'
                  : 'fileName';
          const diff = cmp(key);
          return p.order === 'desc' ? -diff : diff;
        });

        if (!results.length) {
          /* Fallback to filesystem if DB empty */
          const fsList = await fileSvc.listFiles();
          results = (fsList as FileInfo[])
            .filter((f) => !f.isDirectory)
            .map((f) => ({
              fileName: f.name,
              filePath: f.path,
              size: f.size ?? 0,
              mimeType: getMimeType(f.name) || 'application/octet-stream',
              lastModified: f.mtime ?? new Date(),
              createdAt: f.mtime ?? new Date(),
              updatedAt: f.mtime ?? new Date(),
            }));
        }

        const total = results.length;
        const totalPages = Math.ceil(total / p.limit);
        // hasNextPage if more items beyond current page
        const hasNextPage = p.page * p.limit < total;

        // Slice results for current page based on sorted full result set
        const pageData = results.slice((p.page - 1) * p.limit, p.page * p.limit).map((f) => ({
          name: f.fileName,
          path: f.filePath,
          size: f.size,
          mimeType: f.mimeType,
          lastModified: formatDate(f.lastModified),
          url: `/api/files/${encodeURIComponent(f.filePath)}`,
        }));

        // Build pagination links
        const urlObj = new URL(req.url, 'http://localhost');
        urlObj.searchParams.set('page', String(p.page + 1));
        const nextLink = hasNextPage
          ? `${urlObj.pathname}?${urlObj.searchParams.toString()}`
          : null;
        const links = { next: nextLink };

        utils.json(sock, 200, {
          files: pageData,
          pagination: {
            page: p.page,
            limit: p.limit,
            totalFiles: total,
            totalPages,
            hasNextPage,
          },
          _links: links,
        });
      })
      .catch((err) => {
        log.error('LIST failed', { error: err.message });
        utils.plain(sock, 500, 'Failed to list files');
      });
  },

  /* ----------------------------- UPLOAD -------------------------------- */
  async uploadFile(req: IncomingRequest, sock: Socket): Promise<void> {
    if (utils.abortIfClosed(sock)) return;
    const fileName = getQuery(req, 'file') || req.headers['x-filename'];
    if (!fileName) {
      utils.plain(sock, 400, 'Missing file name');
      return;
    }

    await utils
      .withTiming('UPLOAD', { fileName }, async () => {
        const mime = getMimeType(fileName) || '';
        if (!/^image\/|^video\/|^audio\//.test(mime)) {
          utils.plain(sock, 400, 'Only media files allowed');
          return;
        }
        if (!req.body || !(req.body as Buffer).length) {
          utils.plain(sock, 400, 'Empty body');
          return;
        }

        /* Buffer ➜ async iterable (chunked) */
        async function* toChunks(buf: Buffer) {
          const CHUNK = 1 << 20;
          for (let i = 0; i < buf.length; i += CHUNK) yield buf.subarray(i, i + CHUNK);
        }

        await fileSvc.saveFile(fileName, toChunks(req.body as Buffer));

        /* collect + persist stats (non‑blocking on failure) */
        const abs = path.join(config.staticDir, fileName);
        stats
          .getFileStats(abs, config.staticDir)
          .then((meta) => stats.saveFileStats(meta).catch(() => void 0))
          .catch(() => void 0);

        utils.json(sock, 200, { success: true, fileName, size: (req.body as Buffer).length, mime });
      })
      .catch((err) => {
        log.error('UPLOAD failed', { fileName, error: err.message });
        utils.plain(sock, 500, `Upload failed: ${err.message}`);
      });
  },

  /* ----------------------------- DELETE -------------------------------- */
  async deleteFile(req: IncomingRequest, sock: Socket): Promise<void> {
    if (utils.abortIfClosed(sock)) return;
    const fileName = getQuery(req, 'file');
    if (!fileName) {
      utils.plain(sock, 400, 'Missing file name');
      return;
    }

    await utils
      .withTiming('DELETE', { fileName }, async () => {
        await fileSvc.deleteFile(fileName).catch(() => {
          throw new Error('File not found or could not be deleted');
        });
        await stats.deleteFileStats(fileName).catch(() => void 0);
        utils.json(sock, 200, { success: true, fileName, message: 'File deleted' });
      })
      .catch((err) => {
        log.error('DELETE failed', { fileName, error: err.message });
        utils.plain(sock, 404, err.message);
      });
  },

  /* ------------------------- RESOLVE_FILE_NAME ------------------------- */
  resolveFileName(
    req: IncomingRequest,
    sock: Socket,
    extra?: { message: string },
  ): string | undefined {
    const name =
      (req.ctx?.params as Record<string, string>)?.filename ??
      (getQuery(req, 'file') ? decodeURIComponent(getQuery(req, 'file')!) : undefined);

    if (name) return name;

    log.warn('File name missing', { url: req.url, ...(extra ? extra : {}) });
    utils.plain(sock, 400, 'File name is required.');
    return undefined;
  },

  /* ------------------------- RANGE_SENDER_HELPER ----------------------- */
  async sendRange(
    fileName: string,
    stat: Awaited<ReturnType<typeof fileSvc.stat>>,
    mime: string,
    rangeHdr: string,
    sock: Socket,
    acceptEnc: string,
  ) {
    const size = stat.size;
    const [, startStr, endStr] = /bytes=(\d*)-(\d*)/.exec(rangeHdr)!;
    let start = startStr ? +startStr : 0;
    let end = endStr ? +endStr : size - 1;
    if (!startStr) {
      start = Math.max(0, size - end);
      end = size - 1;
    }

    const source = await fileSvc.readFile(fileName, { start, end });
    const len = end - start + 1;

    const headers = {
      'Content-Type': mime,
      'Content-Length': String(len),
      'Content-Range': `bytes ${start}-${end}/${size}`,
      'Accept-Ranges': 'bytes',
      'Accept-Encoding': acceptEnc,
      ...utils.buildCacheHeaders(size, stat.mtime),
    };

    /* no compression for partials (simpler) */
    utils.safeSend(sock, 206, headers, source);
  },
};

/* eof */
