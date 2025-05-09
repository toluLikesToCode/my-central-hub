import { open, Database } from 'sqlite';
import sqlite3 from 'sqlite3';
import { promises as fs } from 'fs';
import path from 'path';
import logger from '../../utils/logger';
import { getMimeType } from '../../utils/helpers';
import { formatDate } from '../../utils/dateFormatter';
import { execFile } from 'child_process';
import { promisify } from 'util';

// Create a logger specific to file stats
const statsLogger = logger.child({
  module: 'file-hosting',
  component: 'stats-helper',
});

const execFileAsync = promisify(execFile);

// Define the file stats structure
export interface FileStats {
  id?: number;
  fileName: string;
  filePath: string;
  mimeType: string;
  size: number;
  lastModified: Date;
  width?: number;
  height?: number;
  duration?: number;
  bitrate?: number;
  encoding?: string;
  codec?: string;
  frameRate?: number;
  audioChannels?: number;
  sampleRate?: number;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * FileHostingStatsHelper
 *
 * Responsible for:
 * - Initializing and maintaining the file_stats.db database
 * - Collecting comprehensive metadata about files
 * - Storing and retrieving file statistics
 * - Supporting queries for file insights and reports
 */
export class FileHostingStatsHelper {
  private db: Database | null = null;
  private dbPath: string;
  private initialized = false;

  /**
   * Constructor for the FileHostingStatsHelper
   * @param dbPath The path to the database file
   */
  constructor(dbPath: string) {
    this.dbPath = dbPath;
    statsLogger.info('File stats helper initialized', { dbPath });
  }

  /**
   * Initialize the database connection and create tables if needed
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Ensure directory exists
      const dir = path.dirname(this.dbPath);
      await fs.mkdir(dir, { recursive: true });

      // Open database connection
      this.db = await open({
        filename: this.dbPath,
        driver: sqlite3.Database,
      });

      // Create the file_stats table if it doesn't exist
      await this.db.exec(`
        CREATE TABLE IF NOT EXISTS file_stats (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          file_name TEXT NOT NULL,
          file_path TEXT NOT NULL UNIQUE,
          mime_type TEXT NOT NULL,
          size INTEGER NOT NULL,
          last_modified DATETIME NOT NULL,
          width INTEGER,
          height INTEGER,
          duration REAL,
          bitrate INTEGER,
          encoding TEXT,
          codec TEXT,
          frame_rate REAL,
          audio_channels INTEGER,
          sample_rate INTEGER,
          created_at DATETIME NOT NULL,
          updated_at DATETIME NOT NULL
        );
        
        CREATE INDEX IF NOT EXISTS idx_file_stats_file_path ON file_stats(file_path);
        CREATE INDEX IF NOT EXISTS idx_file_stats_mime_type ON file_stats(mime_type);
      `);

      this.initialized = true;
      statsLogger.info('Database initialized successfully');
    } catch (error) {
      statsLogger.error('Failed to initialize database', {
        error: (error as Error).message,
        stack: (error as Error).stack,
      });
      throw error;
    }
  }

  /**
   * Close the database connection
   */
  async close(): Promise<void> {
    if (this.db) {
      await this.db.close();
      this.db = null;
      this.initialized = false;
      statsLogger.info('Database connection closed');
    }
  }

  /**
   * Get detailed statistics for a file using appropriate tools based on file type
   * @param filePath Full path to the file
   * @param basePath Base path to make relative paths
   */
  async getFileStats(filePath: string, basePath: string): Promise<FileStats> {
    if (!this.initialized) await this.initialize();

    try {
      const stats = await fs.stat(filePath);
      const relativePath = path.relative(basePath, filePath);
      const fileName = path.basename(filePath);
      const mimeType = getMimeType(fileName) || 'application/octet-stream';

      // Basic file stats that apply to all files
      const fileStats: FileStats = {
        fileName,
        filePath: relativePath,
        mimeType,
        size: stats.size,
        lastModified: stats.mtime,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      // Get extended stats based on file type
      if (mimeType.startsWith('image/')) {
        await this.getImageStats(filePath, fileStats);
      } else if (mimeType.startsWith('video/')) {
        await this.getVideoStats(filePath, fileStats);
      } else if (mimeType.startsWith('audio/')) {
        await this.getAudioStats(filePath, fileStats);
      }

      return fileStats;
    } catch (error) {
      statsLogger.error(`Failed to get stats for file: ${filePath}`, {
        error: (error as Error).message,
      });
      throw error;
    }
  }

  /**
   * Check if statistics exist for a given file path
   * @param filePath Path to the file
   */
  async checkStatsExist(filePath: string): Promise<boolean> {
    if (!this.initialized) await this.initialize();
    if (!this.db) throw new Error('Database not initialized');
    try {
      const result = await this.db.get(
        'SELECT COUNT(*) as count FROM file_stats WHERE file_path = ?',
        [filePath],
      );
      if (result && result.count > 0) {
        return true;
      }
      return false;
    } catch (error) {
      statsLogger.error(`Failed to check stats for file: ${filePath}`, { error: error });
      return false;
    }
  }
  /**
   * Get image-specific statistics
   * @param filePath Path to the image file
   * @param fileStats Stats object to populate
   */
  private async getImageStats(filePath: string, fileStats: FileStats): Promise<void> {
    try {
      // Use ffprobe to get image dimensions
      const { stdout } = await execFileAsync('ffprobe', [
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-show_entries',
        'stream=width,height',
        '-of',
        'csv=p=0',
        filePath,
      ]);

      const [width, height] = stdout.trim().split(',');

      if (width && height) {
        fileStats.width = parseInt(width, 10);
        fileStats.height = parseInt(height, 10);
      }
    } catch (error) {
      statsLogger.warn(`Could not get image dimensions for ${filePath}`, {
        error: (error as Error).message,
      });
      // Continue without image dimensions
    }
  }

  /**
   * Get video-specific statistics
   * @param filePath Path to the video file
   * @param fileStats Stats object to populate
   */
  private async getVideoStats(filePath: string, fileStats: FileStats): Promise<void> {
    try {
      // Use ffprobe to get comprehensive video info
      const { stdout } = await execFileAsync('ffprobe', [
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-show_entries',
        'stream=width,height,codec_name,r_frame_rate,duration,bit_rate',
        '-of',
        'csv=p=0',
        filePath,
      ]);

      const [width, height, codec, frameRate, duration, bitrate] = stdout.trim().split(',');

      if (width) fileStats.width = parseInt(width, 10);
      if (height) fileStats.height = parseInt(height, 10);
      if (codec) fileStats.codec = codec;
      if (duration) fileStats.duration = parseFloat(duration);
      if (bitrate) fileStats.bitrate = parseInt(bitrate, 10);

      // Calculate frame rate from fraction like "30000/1001"
      if (frameRate && frameRate.includes('/')) {
        const [numerator, denominator] = frameRate.split('/').map(Number);
        if (!isNaN(numerator) && !isNaN(denominator) && denominator !== 0) {
          fileStats.frameRate = numerator / denominator;
        }
      }

      // Get audio stream info if available
      try {
        const { stdout: audioStdout } = await execFileAsync('ffprobe', [
          '-v',
          'error',
          '-select_streams',
          'a:0',
          '-show_entries',
          'stream=channels,sample_rate',
          '-of',
          'csv=p=0',
          filePath,
        ]);

        const [channels, sampleRate] = audioStdout.trim().split(',');
        if (channels) fileStats.audioChannels = parseInt(channels, 10);
        if (sampleRate) fileStats.sampleRate = parseInt(sampleRate, 10);
      } catch {
        // Video might not have audio stream, continue
      }
    } catch (error) {
      statsLogger.warn(`Could not get video metadata for ${filePath}`, {
        error: (error as Error).message,
      });
      // Continue without video metadata
    }
  }

  /**
   * Get audio-specific statistics
   * @param filePath Path to the audio file
   * @param fileStats Stats object to populate
   */
  private async getAudioStats(filePath: string, fileStats: FileStats): Promise<void> {
    try {
      // Use ffprobe to get audio information
      const { stdout } = await execFileAsync('ffprobe', [
        '-v',
        'error',
        '-select_streams',
        'a:0',
        '-show_entries',
        'stream=codec_name,channels,sample_rate,duration,bit_rate',
        '-of',
        'csv=p=0',
        filePath,
      ]);

      const [codec, channels, sampleRate, duration, bitrate] = stdout.trim().split(',');

      if (codec) fileStats.codec = codec;
      if (channels) fileStats.audioChannels = parseInt(channels, 10);
      if (sampleRate) fileStats.sampleRate = parseInt(sampleRate, 10);
      if (duration) fileStats.duration = parseFloat(duration);
      if (bitrate) fileStats.bitrate = parseInt(bitrate, 10);
    } catch (error) {
      statsLogger.warn(`Could not get audio metadata for ${filePath}`, {
        error: (error as Error).message,
      });
      // Continue without audio metadata
    }
  }

  /**
   * Store file statistics in the database
   * @param fileStats File statistics to store
   */
  async saveFileStats(fileStats: FileStats): Promise<number> {
    if (!this.initialized) await this.initialize();
    if (!this.db) throw new Error('Database not initialized');

    try {
      const now = formatDate(new Date());

      // Use upsert operation (INSERT OR REPLACE)
      const result = await this.db.run(
        `
        INSERT OR REPLACE INTO file_stats (
          file_name, file_path, mime_type, size, last_modified,
          width, height, duration, bitrate, encoding,
          codec, frame_rate, audio_channels, sample_rate,
          created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `,
        [
          fileStats.fileName,
          fileStats.filePath,
          fileStats.mimeType,
          fileStats.size,
          fileStats.lastModified.toISOString(),
          fileStats.width,
          fileStats.height,
          fileStats.duration,
          fileStats.bitrate,
          fileStats.encoding,
          fileStats.codec,
          fileStats.frameRate,
          fileStats.audioChannels,
          fileStats.sampleRate,
          fileStats.createdAt.toISOString(),
          now,
        ],
      );

      statsLogger.debug(`Stats saved for file: ${fileStats.fileName}`, {
        path: fileStats.filePath,
        size: fileStats.size,
        id: result.lastID,
      });

      return result.lastID || 0;
    } catch (error) {
      statsLogger.error(`Failed to save stats for file: ${fileStats.fileName}`, {
        error: (error as Error).message,
        stack: (error as Error).stack,
      });
      throw error;
    }
  }

  /**
   * Get file statistics from the database by file path
   * @param filePath Relative path to the file
   */
  async getStatsByPath(filePath: string): Promise<FileStats | null> {
    if (!this.initialized) await this.initialize();
    if (!this.db) throw new Error('Database not initialized');

    try {
      const row = await this.db.get(
        `
        SELECT * FROM file_stats WHERE file_path = ?
      `,
        [filePath],
      );

      if (!row) return null;

      return {
        id: row.id,
        fileName: row.file_name,
        filePath: row.file_path,
        mimeType: row.mime_type,
        size: row.size,
        lastModified: new Date(row.last_modified),
        width: row.width,
        height: row.height,
        duration: row.duration,
        bitrate: row.bitrate,
        encoding: row.encoding,
        codec: row.codec,
        frameRate: row.frame_rate,
        audioChannels: row.audio_channels,
        sampleRate: row.sample_rate,
        createdAt: new Date(row.created_at),
        updatedAt: new Date(row.updated_at),
      };
    } catch (error) {
      statsLogger.error(`Failed to retrieve stats for file: ${filePath}`, {
        error: (error as Error).message,
      });
      throw error;
    }
  }

  /**
   * Delete file statistics from the database
   * @param filePath Relative path to the file
   */
  async deleteFileStats(filePath: string): Promise<boolean> {
    if (!this.initialized) await this.initialize();
    if (!this.db) throw new Error('Database not initialized');

    try {
      const result = await this.db.run(
        `
        DELETE FROM file_stats WHERE file_path = ?
      `,
        [filePath],
      );

      const deleted = (result.changes ?? 0) > 0;
      statsLogger.debug(`Stats ${deleted ? 'deleted' : 'not found'} for file: ${filePath}`);
      return deleted;
    } catch (error) {
      statsLogger.error(`Failed to delete stats for file: ${filePath}`, {
        error: (error as Error).message,
      });
      throw error;
    }
  }

  /**
   * Query file statistics with optional filtering
   * @param options Query options for filtering results
   */
  async queryFileStats(
    options: {
      mimeType?: string;
      minSize?: number;
      maxSize?: number;
      hasAudio?: boolean;
      hasVideo?: boolean;
      minWidth?: number;
      minHeight?: number;
      minDuration?: number;
      limit?: number;
      offset?: number;
    } = {},
  ): Promise<FileStats[]> {
    if (!this.initialized) await this.initialize();
    if (!this.db) throw new Error('Database not initialized');

    try {
      const conditions: string[] = [];
      const params: unknown[] = [];

      if (options.mimeType) {
        conditions.push('mime_type LIKE ?');
        params.push(`${options.mimeType}%`);
      }

      if (options.minSize !== undefined) {
        conditions.push('size >= ?');
        params.push(options.minSize);
      }

      if (options.maxSize !== undefined) {
        conditions.push('size <= ?');
        params.push(options.maxSize);
      }

      if (options.hasAudio) {
        conditions.push('audio_channels IS NOT NULL');
      }

      if (options.hasVideo) {
        conditions.push('width IS NOT NULL AND height IS NOT NULL');
      }

      if (options.minWidth !== undefined) {
        conditions.push('width >= ?');
        params.push(options.minWidth);
      }

      if (options.minHeight !== undefined) {
        conditions.push('height >= ?');
        params.push(options.minHeight);
      }

      if (options.minDuration !== undefined) {
        conditions.push('duration >= ?');
        params.push(options.minDuration);
      }

      const whereClause = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';
      const limitClause = options.limit ? `LIMIT ${options.limit}` : '';
      const offsetClause = options.offset ? `OFFSET ${options.offset}` : '';

      const rows = await this.db.all(
        `
        SELECT * FROM file_stats
        ${whereClause}
        ORDER BY last_modified DESC
        ${limitClause}
        ${offsetClause}
      `,
        params,
      );

      return rows.map((row) => ({
        id: row.id,
        fileName: row.file_name,
        filePath: row.file_path,
        mimeType: row.mime_type,
        size: row.size,
        lastModified: new Date(row.last_modified),
        width: row.width,
        height: row.height,
        duration: row.duration,
        bitrate: row.bitrate,
        encoding: row.encoding,
        codec: row.codec,
        frameRate: row.frame_rate,
        audioChannels: row.audio_channels,
        sampleRate: row.sample_rate,
        createdAt: new Date(row.created_at),
        updatedAt: new Date(row.updated_at),
      }));
    } catch (error) {
      statsLogger.error('Failed to query file stats', {
        error: (error as Error).message,
        options,
      });
      throw error;
    }
  }

  /**
   * Get aggregate statistics about files in the database
   */
  async getAggregateStats(): Promise<{
    totalFiles: number;
    totalSize: number;
    avgFileSize: number;
    byMimeType: { mimeType: string; count: number; totalSize: number }[];
    largestFiles: FileStats[];
  }> {
    if (!this.initialized) await this.initialize();
    if (!this.db) throw new Error('Database not initialized');

    try {
      // Get total files and size
      const totals = await this.db.get(`
        SELECT COUNT(*) as total_files, SUM(size) as total_size, AVG(size) as avg_size
        FROM file_stats
      `);

      // Get stats by mime type
      const byMimeType = await this.db.all(`
        SELECT 
          mime_type,
          COUNT(*) as count,
          SUM(size) as total_size
        FROM file_stats
        GROUP BY mime_type
        ORDER BY total_size DESC
      `);

      // Get largest files
      const largestFiles = await this.queryFileStats({
        limit: 10,
      });

      // Sort by size descending
      largestFiles.sort((a, b) => b.size - a.size);

      return {
        totalFiles: totals.total_files || 0,
        totalSize: totals.total_size || 0,
        avgFileSize: totals.avg_size || 0,
        byMimeType: byMimeType.map((row) => ({
          mimeType: row.mime_type,
          count: row.count,
          totalSize: row.total_size,
        })),
        largestFiles: largestFiles.slice(0, 10),
      };
    } catch (error) {
      statsLogger.error('Failed to get aggregate stats', {
        error: (error as Error).message,
      });
      throw error;
    }
  }
}
