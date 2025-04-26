import { createReadStream, statSync, existsSync, readdirSync } from 'fs';
import { Socket } from 'net';
import { join } from 'path';
import { config } from '../../config/server.config';
import { logger } from '../../utils/logger';
import { getMimeType } from '../../utils/helpers';

export class FileService {
  static streamFile(fileName: string, range: string | undefined, socket: Socket) {
    const filePath = join(config.publicDir, 'media', fileName);

    if (!existsSync(filePath)) {
      logger.warn(`File not found: ${fileName}`);
      socket.write('HTTP/1.1 404 Not Found\r\n\r\nFile Not Found');
      socket.end();
      return;
    }

    const fileStat = statSync(filePath);
    const totalSize = fileStat.size;
    const contentType = getMimeType(fileName);

    let start = 0;
    let end = totalSize - 1;

    if (range) {
      const matches = range.match(/bytes=(\d*)-(\d*)/);
      if (matches) {
        const [, startStr, endStr] = matches;
        start = startStr ? parseInt(startStr, 10) : start;
        end = endStr ? parseInt(endStr, 10) : end;
      }
    }

    if (start >= totalSize || end >= totalSize) {
      logger.error(`Invalid range: ${start}-${end}`);
      socket.write('HTTP/1.1 416 Range Not Satisfiable\r\n\r\nInvalid Range');
      socket.end();
      return;
    }

    const chunkSize = end - start + 1;
    const fileStream = createReadStream(filePath, { start, end });

    const headers = [
      'HTTP/1.1 206 Partial Content',
      `Content-Type: ${contentType}`,
      `Content-Length: ${chunkSize}`,
      `Content-Range: bytes ${start}-${end}/${totalSize}`,
      'Accept-Ranges: bytes',
      '',
      '',
    ].join('\r\n');

    socket.write(headers);

    fileStream.pipe(socket);
    fileStream.on('error', (err) => {
      logger.error(`File stream error: ${err.message}`);
      socket.end();
    });
  }

  static listFiles(): { files: string[] } {
    const mediaDir = join(config.publicDir, 'media');

    if (!existsSync(mediaDir)) {
      logger.warn(`Media directory not found: ${mediaDir}`);
      return { files: [] };
    }

    const files = readdirSync(mediaDir).filter(() => {
      // Later: you can filter by allowed extensions (e.g., .mp4, .mp3)
      return true;
    });

    return { files };
  }
}
