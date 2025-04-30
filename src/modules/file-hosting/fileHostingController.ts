// src/modules/file-hosting/fileHostingController.ts
import { Socket } from 'net';
import { sendResponse } from '../../entities/sendResponse';
import { IncomingRequest } from '../../entities/http';
import { FileHostingService } from './fileHostingService';
import { getHeader, getQuery } from '../../utils/httpHelpers';
import { config } from '../../config/server.config';
import { logger } from '../../utils/logger';
import { getMimeType } from '../../utils/helpers';
import { Readable } from 'stream';

const fileSvc = new FileHostingService(config.mediaDir);

export const fileHostingController = {
  /** GET /file?file=filename – serves a file (supports Range) */
  async getFile(req: IncomingRequest, sock: Socket) {
    logger.info(`[getFile] url=${req.url} path=${req.path} query=${JSON.stringify(req.query)}`);
    const fileName = getQuery(req, 'file');
    if (!fileName) {
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
      let stream: Readable;
      const fileStat = await fileSvc.stat(fileName);
      const size = fileStat.size;
      if (rangeHdr) {
        const m = /bytes=(\d*)-(\d*)/.exec(rangeHdr);
        if (!m) {
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
          sendResponse(sock, 416, { 'Content-Type': 'text/plain' }, '416 Range Not Satisfiable');
          sock.end();
          return;
        }
        stream = await fileSvc.readFile(fileName, { start, end });
        if (!stream) throw new Error('Stream is undefined');
        stream.on('error', (err: Error) => {
          logger.error(`[getFile] Stream error: ${err.message}`);
          sendResponse(sock, 500, { 'Content-Type': 'text/plain' }, 'Internal Server Error');
        });
        const len = end - start + 1;
        sendResponse(
          sock,
          206,
          {
            'Content-Type': getMimeType(fileName) || 'application/octet-stream',
            'Accept-Ranges': 'bytes',
            'Content-Range': `bytes ${start}-${end}/${size}`,
            'Content-Length': String(len),
          },
          stream,
        );
      } else {
        stream = await fileSvc.readFile(fileName);
        if (!stream) throw new Error('Stream is undefined');
        stream.on('error', (err: Error) => {
          logger.error(`[getFile] Stream error: ${err.message}`);
          sendResponse(sock, 500, { 'Content-Type': 'text/plain' }, 'Internal Server Error');
        });
        const mimeType = getMimeType(fileName);
        sendResponse(
          sock,
          200,
          { 'Content-Type': mimeType, 'Content-Length': String(size) },
          stream,
        );
      }
    } catch (err: unknown) {
      logger.error(`[getFile] fileName=${fileName}, error=${(err as Error).message}`);
      sendResponse(sock, 404, { 'Content-Type': 'text/plain' }, `File "${fileName}" not found.`);
    }
  },

  /** GET /files – returns JSON list of filenames */
  async listFiles(req: IncomingRequest, sock: Socket) {
    try {
      const files = await fileSvc.listFiles();
      sendResponse(sock, 200, { 'Content-Type': 'application/json' }, JSON.stringify(files));
    } catch (err: unknown) {
      logger.error(`listFiles: ${(err as Error).message}`);
      sendResponse(sock, 500, { 'Content-Type': 'text/plain' }, 'Server error');
    }
  },

  /** POST /file – upload a media file */
  async uploadFile(req: IncomingRequest, sock: Socket) {
    // Assume fileName is provided as a query param or header, and body is the file data
    const fileName = getQuery(req, 'file') || req.headers['x-filename'];
    if (!fileName) {
      sendResponse(sock, 400, { 'Content-Type': 'text/plain' }, 'Missing file name');
      return;
    }
    const mimeType = getMimeType(fileName);
    if (
      !mimeType.startsWith('image/') &&
      !mimeType.startsWith('video/') &&
      !mimeType.startsWith('audio/')
    ) {
      sendResponse(sock, 400, { 'Content-Type': 'text/plain' }, 'Only media files allowed');
      return;
    }
    if (!req.body) {
      sendResponse(sock, 400, { 'Content-Type': 'text/plain' }, 'No file data');
      return;
    }
    // Save file using FileHostingService
    const fileSvc = new FileHostingService(config.mediaDir);
    // Convert Buffer or string to async iterable
    async function* bufferToAsyncIterable(buffer: Buffer) {
      yield buffer;
    }
    const data = typeof req.body === 'string' ? Buffer.from(req.body) : req.body;
    await fileSvc.saveFile(fileName, bufferToAsyncIterable(data));
    sendResponse(sock, 200, { 'Content-Type': 'text/plain' }, 'Upload successful');
  },

  /** DELETE /file?file=filename – delete a media file */
  async deleteFile(req: IncomingRequest, sock: Socket) {
    const fileName = getQuery(req, 'file');
    if (!fileName) {
      sendResponse(sock, 400, { 'Content-Type': 'text/plain' }, 'Missing file name');
      return;
    }
    const fileSvc = new FileHostingService(config.mediaDir);
    const absPath = fileSvc['resolveSafe'](fileName);
    try {
      await import('fs/promises').then((fs) => fs.unlink(absPath));
      sendResponse(sock, 200, { 'Content-Type': 'text/plain' }, 'File deleted');
    } catch {
      sendResponse(
        sock,
        404,
        { 'Content-Type': 'text/plain' },
        'File not found or could not be deleted',
      );
    }
  },
};
