import { Socket } from 'net';
import { sendResponse } from '../../entities/sendResponse';
import { IncomingRequest } from '../../entities/http';
import { FileService } from './fileService';
import { getHeader, getQuery } from '../../utils/httpHelpers';
import { config } from '../../config/server.config';
import { logger } from '../../utils/logger';
import { getMimeType } from '../../utils/helpers';

const fileSvc = new FileService(config.mediaDir);

export const fileStreamingController = {
  /** GET /files – returns JSON list of filenames */
  async listFiles(req: IncomingRequest, sock: Socket) {
    try {
      const files = await fileSvc.listFiles();
      sendResponse(sock, 200, { 'Content-Type': 'application/json' }, JSON.stringify(files));
    } catch (err) {
      logger.error(`listFiles: ${(err as Error).message}`);
      sendResponse(sock, 500, { 'Content-Type': 'text/plain' }, 'Server error');
    }
  },

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
      const rangeHdr = getHeader(req, 'range'); // e.g. "bytes=0-1023"
      let stream,
        start = 0,
        end: number | undefined;

      if (rangeHdr) {
        const [, range] = /bytes=(\d+)-(\d*)/.exec(rangeHdr) ?? [];
        if (!range) throw new Error('Invalid Range header');
        const [s, e] = range.split('-');
        start = Number(s);
        end = e ? Number(e) : undefined;
        stream = await fileSvc.readFile(fileName, { start, end: end ?? start + 1_000_000 });
        const fileStat = await fileSvc.stat(fileName);
        const len = (end ?? fileStat.size - 1) - start + 1;

        sendResponse(
          sock,
          206,
          {
            'Content-Type': 'application/octet-stream',
            'Accept-Ranges': 'bytes',
            'Content-Range': `bytes ${start}-${start + len - 1}/${fileStat.size}`,
            'Content-Length': String(len),
          },
          stream,
        );
      } else {
        stream = await fileSvc.readFile(fileName);
        const fileStat = await fileSvc.stat(fileName);
        const mimeType = getMimeType(fileName);
        sendResponse(
          sock,
          200,
          { 'Content-Type': mimeType, 'Content-Length': String(fileStat.size) },
          stream,
        );
      }
    } catch (err) {
      logger.error(`[handleStream] fileName=${fileName}, error=${(err as Error).message}`);
      sendResponse(sock, 404, { 'Content-Type': 'text/plain' }, `File "${fileName}" not found.`);
    }
  },
};
