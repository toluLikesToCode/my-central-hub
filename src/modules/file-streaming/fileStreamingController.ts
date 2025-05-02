import { Socket } from 'net';
import { sendResponse } from '../../entities/sendResponse';
import { IncomingRequest } from '../../entities/http';
import { getHeader, getQuery } from '../../utils/httpHelpers';
import { config } from '../../config/server.config';
import logger from '../../utils/logger';
import { getMimeType } from '../../utils/helpers';
import { Readable } from 'stream';
import { FileHostingService } from '../file-hosting/fileHostingService';

const fileSvc = new FileHostingService(config.mediaDir);

export const fileStreamingController = {
  /** GET /stream?file=video.mp4 – streams file (supports Range) */
  async handleStream(req: IncomingRequest, sock: Socket) {
    logger.info(
      `[handleStream] url=${req.url} path=${req.path} query=${JSON.stringify(req.query)}`,
    );
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
        const mimeType = getMimeType(fileName);
        sendResponse(
          sock,
          200,
          { 'Content-Type': mimeType, 'Content-Length': String(size) },
          stream,
        );
      }
    } catch (err) {
      logger.error(`[handleStream] fileName=${fileName}, error=${(err as Error).message}`);
      sendResponse(sock, 404, { 'Content-Type': 'text/plain' }, `File "${fileName}" not found.`);
    }
  },
};
