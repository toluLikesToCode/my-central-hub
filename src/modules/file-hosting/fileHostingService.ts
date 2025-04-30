import { createReadStream, createWriteStream } from 'fs';
import { stat, readdir, mkdir } from 'fs/promises';
import { resolve } from 'path';
import { Readable } from 'stream';

export class FileHostingService {
  constructor(private readonly rootDir: string) {}

  private resolveSafe(relPath: string): string {
    const abs = resolve(this.rootDir, relPath);
    if (!abs.startsWith(this.rootDir)) throw new Error('Path traversal attempt');
    return abs;
  }

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
}
