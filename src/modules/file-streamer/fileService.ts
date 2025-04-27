// file-streamer/fileService.ts
import {
  existsSync,
  createReadStream,
  statSync,
  createWriteStream,
  readdirSync, // ← added for legacy tests
} from 'fs';
import { stat, readdir, mkdir } from 'fs/promises';
import { resolve, join } from 'path';
import { Readable } from 'stream';
import { Socket } from 'net';
import { config } from '../../config/server.config';
import { getMimeType } from '../../utils/helpers';
import { sendResponse } from '../../entities/sendResponse';

export class FileService {
  constructor(private readonly rootDir: string) {}

  private resolveSafe(relPath: string): string {
    const abs = resolve(this.rootDir, relPath);
    if (!abs.startsWith(this.rootDir)) throw new Error('Path traversal attempt');
    return abs;
  }

  /* ---------- modern async API ---------- */

  async listFiles(relDir = '.'): Promise<string[]> {
    return await readdir(this.resolveSafe(relDir));
  }

  async stat(relPath: string) {
    return await stat(this.resolveSafe(relPath));
  }

  async readFile(relPath: string, range?: { start: number; end: number }): Promise<Readable> {
    const abs = this.resolveSafe(relPath);
    return createReadStream(abs, range);
  }

  async saveFile(relPath: string, data: AsyncIterable<Buffer>): Promise<void> {
    const abs = this.resolveSafe(relPath);
    await mkdir(resolve(abs, '..'), { recursive: true });
    const ws = createWriteStream(abs);
    for await (const chunk of data) ws.write(chunk);
    await new Promise<void>((res, rej) => {
      ws.end(res);
      ws.on('error', rej);
    });
  }

  /* ---------- legacy static helpers (keep old tests green) ---------- */

  static listFiles(): { files: string[] } {
    const dir = config.mediaDir;
    if (!existsSync(dir)) return { files: [] };
    return { files: readdirSync(dir) };
  }

  static streamFile(filename: string, rangeHeader: string | undefined, socket: Socket): void {
    const abs = join(config.mediaDir, filename);

    if (!existsSync(abs)) {
      sendResponse(socket, 404, { 'Content-Type': 'text/plain' }, '404 Not Found');
      socket.end();
      return;
    }

    const stats = statSync(abs);
    const size = stats.size;

    let start = 0;
    let end = size - 1;

    if (rangeHeader) {
      const m = /bytes=(\d+)-(\d*)/.exec(rangeHeader);
      if (!m) {
        sendResponse(socket, 416, { 'Content-Type': 'text/plain' }, '416 Range Not Satisfiable');
        socket.end();
        return;
      }
      start = Number(m[1]);
      end = m[2] ? Number(m[2]) : end;

      if (start > end || start >= size) {
        sendResponse(socket, 416, { 'Content-Type': 'text/plain' }, '416 Range Not Satisfiable');
        socket.end();
        return;
      }
    }

    const len = end - start + 1;
    const stream = createReadStream(abs, { start, end });
    const headers = {
      'Content-Type': getMimeType(filename) ?? 'application/octet-stream',
      'Accept-Ranges': 'bytes',
      'Content-Range': `bytes ${start}-${end}/${size}`,
      'Content-Length': String(len),
    };

    sendResponse(socket, 206, headers, stream);

    stream.on('error', () => socket.end());
  }
  /* ------------------------------------------------------------------ */
}
