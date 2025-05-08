import { createReadStream, createWriteStream } from 'fs';
import { stat, readdir, mkdir, unlink, readFile } from 'fs/promises';
import { resolve } from 'path';
import { Readable } from 'stream';
import { getMimeType } from '../../utils/helpers';
import logger from '../../utils/logger';
import { LRUCache } from 'lru-cache';
import type { LRUCache as LRUCacheType } from 'lru-cache';
import { PassThrough } from 'stream';
import { config } from '../../config/server.config';

// Create a logger specific to the file hosting service
const fileServiceLogger = logger.child({
  module: 'file-hosting',
  component: 'service',
});

// Define file cache options and types
interface CachedFile {
  buffer: Buffer;
  mtime: number;
  size: number;
  mimeType: string;
}

interface CacheOptions {
  maxSize: number; // Maximum size in bytes
  maxAge: number; // Maximum age in milliseconds
  enabled: boolean; // Whether caching is enabled
}

// Get cache options from config
const DEFAULT_CACHE_OPTIONS: CacheOptions = {
  maxSize: config.fileCache?.maxSize || 200 * 1024 * 1024, // 200MB
  maxAge: config.fileCache?.maxAge || 10 * 60 * 1000, // 10 minutes
  enabled: config.fileCache?.enabled !== false, // Default to true
};

/**
 * File Hosting Service
 *
 * Provides methods for listing, reading, saving, and deleting files
 * with optional caching for better performance of small and medium files.
 */
export class FileHostingService {
  private fileCache: LRUCacheType<string, CachedFile>;
  private cacheOptions: CacheOptions;
  private cacheStats = {
    hits: 0,
    misses: 0,
    size: 0,
  };

  constructor(
    private readonly rootDir: string,
    cacheOptions?: Partial<CacheOptions>,
  ) {
    // Merge provided options with defaults
    this.cacheOptions = {
      ...DEFAULT_CACHE_OPTIONS,
      ...cacheOptions,
    };

    // Initialize LRU cache with size and TTL limits
    this.fileCache = new LRUCache<string, CachedFile>({
      maxSize: this.cacheOptions.maxSize,
      ttl: this.cacheOptions.maxAge,
      // Calculate size of cache entries
      sizeCalculation: (value: CachedFile) => value.buffer.length,
      // Add cache statistics for monitoring
      dispose: () => {
        this.cacheStats.size = this.fileCache.calculatedSize || 0;
      },
    });

    fileServiceLogger.info('File hosting service initialized', {
      cacheEnabled: this.cacheOptions.enabled,
      maxCacheSize: `${Math.round(this.cacheOptions.maxSize / (1024 * 1024))} MB`,
      cacheTTL: `${Math.round(this.cacheOptions.maxAge / 60000)} minutes`,
    });
  }

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

  /**
   * Retrieves a file as a readable stream, with optional caching for small/medium files
   * Caching is skipped for range requests and large files
   */
  async readFile(relPath: string, range?: { start: number; end: number }): Promise<Readable> {
    const abs = this.resolveSafe(relPath);

    // Use direct stream for range requests (skip cache)
    if (range || !this.cacheOptions.enabled) {
      return createReadStream(abs, range);
    }

    try {
      // Check if cache is enabled and the file is in cache
      const fileStat = await stat(abs);
      const fileSize = fileStat.size;
      const mimeType = getMimeType(relPath) || 'application/octet-stream';

      // Only cache files under 10MB
      const CACHE_SIZE_LIMIT = 10 * 1024 * 1024; // 10MB
      const shouldCache = fileSize <= CACHE_SIZE_LIMIT;

      // If file is in cache and up-to-date, return from cache
      const cached = this.fileCache.get(relPath);
      if (cached && cached.mtime === fileStat.mtime.getTime()) {
        this.cacheStats.hits++;

        fileServiceLogger.debug('Cache hit', {
          relPath,
          size: fileSize,
          cacheHits: this.cacheStats.hits,
          cacheMisses: this.cacheStats.misses,
        });

        // Convert buffer to stream for consistent API
        const stream = new PassThrough();
        stream.end(cached.buffer);
        return stream;
      }

      // Cache miss, load from disk
      this.cacheStats.misses++;

      if (shouldCache) {
        // Read full file into memory and cache it
        const buffer = await readFile(abs);
        this.fileCache.set(relPath, {
          buffer,
          mtime: fileStat.mtime.getTime(),
          size: fileSize,
          mimeType,
        });

        this.cacheStats.size = this.fileCache.calculatedSize || 0;

        fileServiceLogger.debug('File cached', {
          relPath,
          size: fileSize,
          cacheSize: `${Math.round(this.cacheStats.size / (1024 * 1024))} MB`,
          totalCached: this.fileCache.size,
        });

        // Return a stream from the buffer for consistent API
        const stream = new PassThrough();
        stream.end(buffer);
        return stream;
      } else {
        // Too large to cache, use direct stream
        fileServiceLogger.debug('File too large to cache, streaming directly', {
          relPath,
          size: fileSize,
          limit: CACHE_SIZE_LIMIT,
        });
        return createReadStream(abs);
      }
    } catch (error) {
      fileServiceLogger.error('Error reading file', {
        error: (error as Error).message,
        path: abs,
      });
      throw error;
    }
  }

  /**
   * Get cache statistics for monitoring
   */
  getCacheStats() {
    return {
      ...this.cacheStats,
      size: this.fileCache.calculatedSize || 0,
      itemCount: this.fileCache.size,
      maxSize: this.cacheOptions.maxSize,
      hitRatio: this.cacheStats.hits / (this.cacheStats.hits + this.cacheStats.misses) || 0,
    };
  }

  /**
   * Clear the entire file cache
   */
  clearCache() {
    const itemCount = this.fileCache.size;
    this.fileCache.clear();
    this.cacheStats.size = 0;
    fileServiceLogger.info('Cache cleared', { itemCount });
    return { cleared: itemCount };
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

    // Invalidate cache if this file was in it
    if (this.fileCache.has(relPath)) {
      this.fileCache.delete(relPath);
      fileServiceLogger.debug('Cache entry invalidated after update', { relPath });
    }
  }

  async deleteFile(relPath: string): Promise<void> {
    const abs = this.resolveSafe(relPath);
    await unlink(abs);

    // Remove from cache if present
    if (this.fileCache.has(relPath)) {
      this.fileCache.delete(relPath);
      fileServiceLogger.debug('Cache entry removed after file deletion', { relPath });
    }
  }
}
