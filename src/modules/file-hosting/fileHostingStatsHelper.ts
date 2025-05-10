/* eslint-disable @typescript-eslint/no-explicit-any */
import { open, Database } from 'sqlite';
import sqlite3 from 'sqlite3';
import { promises as fs } from 'fs';
import path from 'path';
import logger from '../../utils/logger'; // Assuming this path is correct relative to your project structure
import { getMimeType } from '../../utils/helpers'; // Assuming this path is correct
import { execFile } from 'child_process';
import { promisify } from 'util';

// Create a logger specific to file stats
const statsLogger = logger.child({
  module: 'file-hosting',
  component: 'stats-helper',
});

const execFileAsync = promisify(execFile);

// Define the file stats structure (public interface)
export interface FileStats {
  id?: number;
  fileName: string;
  filePath: string; // Relative path
  mimeType: string;
  size: number; // in bytes
  lastModified: Date;
  width?: number; // pixels
  height?: number; // pixels
  duration?: number; // seconds
  bitrate?: number; // bits per second
  encoding?: string; // e.g., codec long name, pixel format
  codec?: string; // e.g., h264, mp3
  frameRate?: number; // fps
  audioChannels?: number;
  sampleRate?: number; // Hz
  createdAt: Date;
  updatedAt: Date;
}

// Internal type for database rows
interface FileStatsDbRow {
  id: number;
  file_name: string;
  file_path: string;
  mime_type: string;
  size: number;
  last_modified: string; // ISO Date String
  width: number | null;
  height: number | null;
  duration: number | null;
  bitrate: number | null;
  encoding: string | null;
  codec: string | null;
  frame_rate: number | null;
  audio_channels: number | null;
  sample_rate: number | null;
  created_at: string; // ISO Date String
  updated_at: string; // ISO Date String
}

// Helper functions for safe parsing of ffprobe output
function safeParseInt(value: any, radix: number = 10): number | undefined {
  if (value === null || value === undefined || String(value).toUpperCase() === 'N/A') {
    return undefined;
  }
  const parsed = parseInt(String(value), radix);
  return isNaN(parsed) ? undefined : parsed;
}

function safeParseFloat(value: any): number | undefined {
  if (value === null || value === undefined || String(value).toUpperCase() === 'N/A') {
    return undefined;
  }
  const parsed = parseFloat(String(value));
  return isNaN(parsed) ? undefined : parsed;
}

function parseFrameRate(frameRateStr?: string): number | undefined {
  if (!frameRateStr) return undefined;
  if (frameRateStr.includes('/')) {
    const [numStr, denStr] = frameRateStr.split('/');
    const num = parseInt(numStr, 10);
    const den = parseInt(denStr, 10);
    if (!isNaN(num) && !isNaN(den) && den !== 0) {
      return num / den;
    }
  } else {
    const fr = safeParseFloat(frameRateStr);
    return fr && fr > 0 ? fr : undefined;
  }
  return undefined;
}

// Interfaces for ffprobe JSON output structure
interface FFProbeStream {
  index: number;
  codec_name?: string;
  codec_long_name?: string;
  codec_type?: 'video' | 'audio' | 'subtitle' | 'data' | 'attachment';
  width?: number;
  height?: number;
  duration?: string;
  bit_rate?: string;
  r_frame_rate?: string;
  channels?: number;
  channel_layout?: string;
  sample_rate?: string;
  pix_fmt?: string;
  // Other stream properties can be added if needed
}

interface FFProbeFormat {
  filename: string;
  nb_streams: number;
  nb_programs: number;
  format_name: string;
  format_long_name: string;
  start_time?: string;
  duration?: string;
  size?: string;
  bit_rate?: string;
  probe_score?: number;
  tags?: Record<string, string>;
  // Other format properties can be added if needed
}

interface FFProbeOutput {
  streams: FFProbeStream[];
  format: FFProbeFormat;
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
   * @param dbPath The path to the SQLite database file
   */
  constructor(dbPath: string) {
    this.dbPath = dbPath;
    statsLogger.info('FileHostingStatsHelper instance created.', { dbPath });
  }

  /**
   * Initialize the database connection and create tables if they don't exist.
   */
  async initialize(): Promise<void> {
    if (this.initialized && this.db) {
      statsLogger.debug('Database already initialized.');
      return;
    }

    try {
      const dir = path.dirname(this.dbPath);
      await fs.mkdir(dir, { recursive: true });

      this.db = await open({
        filename: this.dbPath,
        driver: sqlite3.Database,
      });

      // WAL mode can improve concurrency and performance
      await this.db.exec('PRAGMA journal_mode = WAL;');

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
          duration REAL,        -- Store duration in seconds
          bitrate INTEGER,       -- Store bitrate in bps
          encoding TEXT,
          codec TEXT,
          frame_rate REAL,
          audio_channels INTEGER,
          sample_rate INTEGER,
          created_at DATETIME NOT NULL,
          updated_at DATETIME NOT NULL
        );
      `);

      // Ensure all desired indexes exist
      const indexes = [
        { name: 'idx_file_stats_file_path', column: 'file_path' },
        { name: 'idx_file_stats_mime_type', column: 'mime_type' },
        { name: 'idx_file_stats_size', column: 'size' },
        { name: 'idx_file_stats_last_modified', column: 'last_modified' },
        { name: 'idx_file_stats_duration', column: 'duration' },
      ];

      for (const index of indexes) {
        await this.db.exec(
          `CREATE INDEX IF NOT EXISTS ${index.name} ON file_stats(${index.column});`,
        );
      }

      this.initialized = true;
      statsLogger.info('Database initialized successfully.', { dbPath: this.dbPath });
    } catch (error) {
      statsLogger.error('Failed to initialize database.', {
        dbPath: this.dbPath,
        error: (error as Error).message,
        stack: (error as Error).stack,
      });
      this.initialized = false; // Ensure not marked as initialized on failure
      this.db = null;
      throw error; // Re-throw to indicate initialization failure
    }
  }

  /**
   * Close the database connection.
   */
  async close(): Promise<void> {
    if (this.db) {
      try {
        await this.db.close();
        statsLogger.info('Database connection closed.');
      } catch (error) {
        statsLogger.error('Error closing database connection.', {
          error: (error as Error).message,
        });
      } finally {
        this.db = null;
        this.initialized = false;
      }
    }
  }

  private async getMediaInfo(filePath: string): Promise<FFProbeOutput | null> {
    try {
      const { stdout } = await execFileAsync('ffprobe', [
        '-v',
        'error', // Only log errors
        '-show_format', // Get format information
        '-show_streams', // Get stream information
        '-print_format',
        'json', // Output as JSON
        filePath,
      ]);
      return JSON.parse(stdout) as FFProbeOutput;
    } catch (error: any) {
      // Check for common ffprobe errors, e.g., file not found or not a media file
      const errorMessage = error.stderr || error.message || 'Unknown ffprobe error';
      errorMessage.includes('No such file or directory'); // just so i dont get a warning
      statsLogger.warn(`ffprobe execution failed for: ${path.basename(filePath)}`, {
        filePath,
        //error: errorMessage, // Can be very verbose
      });
      return null;
    }
  }

  /**
   * Get detailed statistics for a file.
   * @param filePath Full path to the file.
   * @param basePath Base path to make the stored file path relative.
   */
  async getFileStats(filePath: string, basePath: string): Promise<FileStats> {
    if (!this.initialized || !this.db) {
      await this.initialize(); // Ensure initialized
      if (!this.db) throw new Error('Database initialization failed or not complete.');
    }

    try {
      const fileSystemStats = await fs.stat(filePath);
      if (!fileSystemStats.isFile()) {
        throw new Error(`Path is not a file: ${filePath}`);
      }

      const relativePath = path.relative(basePath, filePath);
      const fileName = path.basename(filePath);
      const mimeType = getMimeType(fileName) || 'application/octet-stream'; // Fallback MIME type

      const baseFileStats: FileStats = {
        fileName,
        filePath: relativePath,
        mimeType,
        size: fileSystemStats.size,
        lastModified: fileSystemStats.mtime,
        createdAt: new Date(), // Will be overridden if stats already exist and we are just fetching
        updatedAt: new Date(), // Will be set on save/update
      };

      const mediaInfo = await this.getMediaInfo(filePath);

      if (mediaInfo) {
        // Populate common stats first, then type-specific
        if (mediaInfo.format) {
          baseFileStats.duration = safeParseFloat(mediaInfo.format.duration);
          baseFileStats.bitrate = safeParseInt(mediaInfo.format.bit_rate);
        }

        if (mimeType.startsWith('image/')) {
          this.populateImageStats(mediaInfo, baseFileStats);
        } else if (mimeType.startsWith('video/')) {
          this.populateVideoStats(mediaInfo, baseFileStats);
        } else if (mimeType.startsWith('audio/')) {
          this.populateAudioStats(mediaInfo, baseFileStats);
        }
      }
      return baseFileStats;
    } catch (error) {
      statsLogger.error(`Failed to get file stats for: ${filePath}`, {
        error: (error as Error).message,
        stack: (error as Error).stack,
      });
      throw error;
    }
  }

  private populateImageStats(mediaInfo: FFProbeOutput, fileStats: FileStats): void {
    const imageStream = mediaInfo.streams.find((s) => s.codec_type === 'video');
    if (imageStream) {
      fileStats.width = safeParseInt(imageStream.width);
      fileStats.height = safeParseInt(imageStream.height);
      fileStats.codec = imageStream.codec_name;
      fileStats.encoding = imageStream.pix_fmt || imageStream.codec_long_name; // Prefer pixel format for images

      // Bitrate for single images is often not applicable from stream, use format if available
      if (fileStats.bitrate === undefined) {
        // if not set by format
        fileStats.bitrate = safeParseInt(imageStream.bit_rate);
      }
    }
  }

  private populateVideoStats(mediaInfo: FFProbeOutput, fileStats: FileStats): void {
    const videoStream = mediaInfo.streams.find((s) => s.codec_type === 'video');
    const audioStream = mediaInfo.streams.find((s) => s.codec_type === 'audio');

    if (videoStream) {
      fileStats.width = safeParseInt(videoStream.width);
      fileStats.height = safeParseInt(videoStream.height);
      fileStats.codec = videoStream.codec_name;
      fileStats.encoding = videoStream.pix_fmt || videoStream.codec_long_name;
      fileStats.frameRate = parseFrameRate(videoStream.r_frame_rate);
      // Prefer video stream duration if format duration was not available or less precise
      const streamDuration = safeParseFloat(videoStream.duration);
      if (
        streamDuration !== undefined &&
        (fileStats.duration === undefined ||
          Math.abs(streamDuration - (fileStats.duration || 0)) > 0.001)
      ) {
        fileStats.duration = streamDuration;
      }
      // Prefer video stream bitrate if format bitrate was not specific enough
      const videoBitrate = safeParseInt(videoStream.bit_rate);
      if (videoBitrate !== undefined && (fileStats.bitrate === undefined || videoBitrate > 0)) {
        // Ensure it's a valid bitrate
        fileStats.bitrate = videoBitrate;
      }
    }

    if (audioStream) {
      fileStats.audioChannels = safeParseInt(audioStream.channels);
      fileStats.sampleRate = safeParseInt(audioStream.sample_rate);
      // Note: Main codec/encoding/bitrate on FileStats is for video.
      // For dedicated audio stats, extend FileStats or store in a related table.
    }
    // Ensure overall format duration and bitrate (already set in getFileStats) take precedence if more accurate or available.
    // The logic in getFileStats already sets format.duration and format.bit_rate initially.
    // Here we are refining with stream specific data if format data was lacking or less specific.
  }

  private populateAudioStats(mediaInfo: FFProbeOutput, fileStats: FileStats): void {
    const audioStream = mediaInfo.streams.find((s) => s.codec_type === 'audio');
    if (audioStream) {
      fileStats.codec = audioStream.codec_name;
      fileStats.encoding = audioStream.codec_long_name;
      fileStats.audioChannels = safeParseInt(audioStream.channels);
      fileStats.sampleRate = safeParseInt(audioStream.sample_rate);

      const streamDuration = safeParseFloat(audioStream.duration);
      if (
        streamDuration !== undefined &&
        (fileStats.duration === undefined ||
          Math.abs(streamDuration - (fileStats.duration || 0)) > 0.001)
      ) {
        fileStats.duration = streamDuration;
      }

      const audioBitrate = safeParseInt(audioStream.bit_rate);
      if (audioBitrate !== undefined && (fileStats.bitrate === undefined || audioBitrate > 0)) {
        fileStats.bitrate = audioBitrate;
      }
    }
    // Similar to video, format level duration/bitrate from getFileStats are primary.
  }

  /**
   * Store or update file statistics in the database.
   * @param fileStats File statistics to store.
   * @returns The ID of the inserted or updated record.
   */
  async saveFileStats(fileStats: FileStats): Promise<number> {
    if (!this.initialized || !this.db) {
      await this.initialize();
      if (!this.db) throw new Error('Database initialization failed or not complete.');
    }

    const now = new Date();
    const statsToSave = {
      ...fileStats,
      lastModified:
        fileStats.lastModified instanceof Date
          ? fileStats.lastModified.toISOString()
          : new Date(fileStats.lastModified).toISOString(),
      createdAt: (fileStats.createdAt instanceof Date
        ? fileStats.createdAt
        : new Date(fileStats.createdAt || now)
      ).toISOString(),
      updatedAt: now.toISOString(),
    };

    try {
      const result = await this.db.run(
        `INSERT INTO file_stats (
          file_name, file_path, mime_type, size, last_modified,
          width, height, duration, bitrate, encoding, codec,
          frame_rate, audio_channels, sample_rate, created_at, updated_at
        ) VALUES (
          $fileName, $filePath, $mimeType, $size, $lastModified,
          $width, $height, $duration, $bitrate, $encoding, $codec,
          $frameRate, $audioChannels, $sampleRate, $createdAt, $updatedAt
        )
        ON CONFLICT(file_path) DO UPDATE SET
          file_name = excluded.file_name,
          mime_type = excluded.mime_type,
          size = excluded.size,
          last_modified = excluded.last_modified,
          width = excluded.width,
          height = excluded.height,
          duration = excluded.duration,
          bitrate = excluded.bitrate,
          encoding = excluded.encoding,
          codec = excluded.codec,
          frame_rate = excluded.frame_rate,
          audio_channels = excluded.audio_channels,
          sample_rate = excluded.sample_rate,
          updated_at = excluded.updated_at`,
        {
          $fileName: statsToSave.fileName,
          $filePath: statsToSave.filePath,
          $mimeType: statsToSave.mimeType,
          $size: statsToSave.size,
          $lastModified: statsToSave.lastModified,
          $width: statsToSave.width,
          $height: statsToSave.height,
          $duration: statsToSave.duration,
          $bitrate: statsToSave.bitrate,
          $encoding: statsToSave.encoding,
          $codec: statsToSave.codec,
          $frameRate: statsToSave.frameRate,
          $audioChannels: statsToSave.audioChannels,
          $sampleRate: statsToSave.sampleRate,
          $createdAt: statsToSave.createdAt, // This is correct due to ON CONFLICT not re-running excluded.created_at
          $updatedAt: statsToSave.updatedAt,
        },
      );

      let recordId = result.lastID;
      if (!recordId && result.changes && result.changes > 0) {
        // UPSERT resulted in an UPDATE, fetch the ID
        const updatedRecord = await this.getStatsByPath(fileStats.filePath);
        recordId = updatedRecord?.id;
      }

      statsLogger.debug(`Stats saved for ${fileStats.filePath}`, {
        id: recordId,
        changes: result.changes,
      });
      return recordId || 0; // Return 0 if ID couldn't be determined (should not happen if save was successful)
    } catch (error) {
      statsLogger.error(`Failed to save stats for ${fileStats.filePath}`, {
        error: (error as Error).message,
        stack: (error as Error).stack,
      });
      throw error;
    }
  }

  /**
   * Get file statistics from the database by relative file path.
   * @param filePath Relative path to the file.
   */
  async getStatsByPath(filePath: string): Promise<FileStats | null> {
    if (!this.initialized || !this.db) {
      await this.initialize();
      if (!this.db) throw new Error('Database initialization failed or not complete.');
    }

    try {
      const row = await this.db.get<FileStatsDbRow>(
        'SELECT * FROM file_stats WHERE file_path = ?',
        filePath,
      );
      return row ? this.mapDbRowToFileStats(row) : null;
    } catch (error) {
      statsLogger.error(`Failed to get stats by path: ${filePath}`, {
        error: (error as Error).message,
      });
      throw error;
    }
  }

  /**
   * Check if statistics exist for a given file path.
   * @param filePath Path to the file (relative path expected for DB lookup).
   */
  async checkStatsExist(filePath: string): Promise<boolean> {
    if (!this.initialized || !this.db) {
      await this.initialize();
      if (!this.db) throw new Error('Database initialization failed or not complete.');
    }
    try {
      const result = await this.db.get<{ count: number }>(
        'SELECT COUNT(*) as count FROM file_stats WHERE file_path = ?',
        [filePath],
      );
      return !!(result && result.count > 0);
    } catch (error) {
      statsLogger.error(`Failed to check if stats exist for: ${filePath}`, {
        error: (error as Error).message,
      });
      return false; // Return false on error to avoid downstream issues
    }
  }

  /**
   * Delete file statistics from the database by relative file path.
   * @param filePath Relative path to the file.
   */
  async deleteFileStats(filePath: string): Promise<boolean> {
    if (!this.initialized || !this.db) {
      await this.initialize();
      if (!this.db) throw new Error('Database initialization failed or not complete.');
    }
    try {
      const result = await this.db.run('DELETE FROM file_stats WHERE file_path = ?', filePath);
      statsLogger.debug(`Delete operation for ${filePath}`, { changes: result.changes });
      return (result.changes ?? 0) > 0;
    } catch (error) {
      statsLogger.error(`Failed to delete stats for ${filePath}`, {
        error: (error as Error).message,
      });
      throw error;
    }
  }

  // Method to be replaced in your FileHostingStatsHelper.ts class

  /**
   * Query file statistics with optional filtering, sorting, and pagination.
   * This method applies simple filters at the database level.
   * Complex filtering (and/or/not/regex) is expected to be handled
   * by the calling layer (e.g., FileHostingController) after an initial data fetch.
   */
  async queryFileStats(
    options: {
      // Simple filters based on direct properties
      mimeType?: string; // Filter by MIME type prefix (e.g., "image/", "video/mp4")
      minSize?: number;
      maxSize?: number;
      hasAudio?: boolean; // True if file should have audio channels
      hasVideo?: boolean; // True if file should have width and height
      minWidth?: number;
      minHeight?: number;
      minDuration?: number; // In seconds

      // Pagination and Sorting
      limit?: number;
      offset?: number;
      sortBy?: keyof FileStats; // Property of FileStats to sort by
      sortOrder?: 'ASC' | 'DESC';

      // Note: This method does NOT process complex nested filter objects
      // (like 'and', 'or', 'not', 'regex' from the FileFilter interface)
      // directly into its SQL query. Those are handled by the controller.
      // If such properties are present in `options` due to spreading, they are ignored here.
    } = {},
  ): Promise<FileStats[]> {
    if (!this.initialized || !this.db) {
      await this.initialize(); // Ensure DB is initialized
      if (!this.db) {
        statsLogger.error('Database not available for queryFileStats.');
        throw new Error('Database initialization failed or not complete.');
      }
    }

    const params: any[] = [];
    const conditions: string[] = [];

    // Build WHERE clause from simple, top-level options
    if (options.mimeType) {
      // Handle common cases: exact match or prefix match
      if (options.mimeType.endsWith('/')) {
        conditions.push('mime_type LIKE ?');
        params.push(`${options.mimeType}%`); // e.g., "image/%"
      } else if (options.mimeType.includes('/')) {
        conditions.push('mime_type = ?'); // e.g., "image/jpeg"
        params.push(options.mimeType);
      } else {
        // If just "image" or "video", treat as prefix for the major type
        conditions.push('mime_type LIKE ?');
        params.push(`${options.mimeType}/%`);
      }
    }
    if (options.minSize !== undefined) {
      conditions.push('size >= ?');
      params.push(options.minSize);
    }
    if (options.maxSize !== undefined && options.maxSize > 0) {
      // Ensure maxSize is meaningful
      conditions.push('size <= ?');
      params.push(options.maxSize);
    }
    if (options.hasAudio) {
      conditions.push('audio_channels IS NOT NULL AND audio_channels > 0');
    }
    if (options.hasVideo) {
      conditions.push('(width IS NOT NULL AND width > 0) AND (height IS NOT NULL AND height > 0)');
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

    const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';

    // Sorting logic (maps FileStats camelCase keys to DB snake_case columns)
    const validSortKeys = [
      'id',
      'fileName',
      'filePath',
      'mimeType',
      'size',
      'lastModified',
      'width',
      'height',
      'duration',
      'bitrate',
      'encoding',
      'codec',
      'frameRate',
      'audioChannels',
      'sampleRate',
      'createdAt',
      'updatedAt',
    ] as const;
    type ValidSortKey = (typeof validSortKeys)[number];

    const sortKeyMap: Record<ValidSortKey, string> = {
      id: 'id',
      fileName: 'file_name',
      filePath: 'file_path',
      mimeType: 'mime_type',
      size: 'size',
      lastModified: 'last_modified',
      width: 'width',
      height: 'height',
      duration: 'duration',
      bitrate: 'bitrate',
      encoding: 'encoding',
      codec: 'codec',
      frameRate: 'frame_rate',
      audioChannels: 'audio_channels',
      sampleRate: 'sample_rate',
      createdAt: 'created_at',
      updatedAt: 'updated_at',
    };

    const sortByKey: ValidSortKey =
      options.sortBy && (validSortKeys as readonly string[]).includes(options.sortBy)
        ? options.sortBy
        : 'fileName'; // Default sort to fileName if not specified
    const dbSortColumn = sortKeyMap[sortByKey];
    const sortOrder = options.sortOrder?.toUpperCase() === 'ASC' ? 'ASC' : 'DESC'; // Default to DESC if invalid
    // Add secondary sort by ID for stable pagination, especially when primary sort key might have duplicates
    const orderByClause = `ORDER BY "${dbSortColumn}" ${sortOrder}, "id" ${sortOrder}`;

    // Pagination
    // The controller might pass a large limit (MAX_RECORDS_FOR_JS_FILTERING) or a smaller one.
    const limit = Math.max(1, safeParseInt(options.limit, 10) || 20); // Default limit
    const offset = Math.max(0, safeParseInt(options.offset, 10) || 0); // Default offset

    const sqlQuery = `
      SELECT * FROM file_stats
      ${whereClause}
      ${orderByClause}
      LIMIT ? OFFSET ?
    `;
    const finalParams = [...params, limit, offset];

    statsLogger.debug('Executing queryFileStats SQL', {
      query: sqlQuery,
      params: finalParams,
      originalOptions: options,
    });

    try {
      const rows = await this.db.all<FileStatsDbRow[]>(sqlQuery, finalParams);
      return rows.map(this.mapDbRowToFileStats.bind(this)); // Use bind if mapDbRowToFileStats uses `this`
    } catch (error) {
      statsLogger.error('Failed to query file stats from database', {
        error: (error as Error).message,
        stack: (error as Error).stack,
        sql: sqlQuery,
        params: finalParams,
      });
      throw error; // Re-throw the error to be handled by the controller
    }
  }

  /**
   * Get aggregate statistics about all files in the database.
   */
  async getAggregateStats(): Promise<{
    totalFiles: number;
    totalSize: number;
    avgFileSize: number;
    byMimeType: Array<{ mimeType: string; count: number; totalSize: number; avgSize: number }>;
    largestFiles: FileStats[];
  }> {
    if (!this.initialized || !this.db) {
      await this.initialize();
      if (!this.db) throw new Error('Database initialization failed or not complete.');
    }

    try {
      const totals = await this.db.get<{
        total_files: number;
        total_size: number;
        avg_file_size: number;
      }>(
        'SELECT COUNT(*) as total_files, SUM(size) as total_size, AVG(size) as avg_file_size FROM file_stats',
      );

      const mimeTypeAggregates = await this.db.all<
        { mime_type: string; file_count: number; sum_size: number; avg_mime_size: number }[]
      >(`SELECT mime_type, COUNT(*) as file_count, SUM(size) as sum_size, AVG(size) as avg_mime_size 
          FROM file_stats 
          GROUP BY mime_type 
          ORDER BY file_count DESC`);

      const largestFiles = await this.queryFileStats({
        sortBy: 'size',
        sortOrder: 'DESC',
        limit: 10,
      });

      return {
        totalFiles: totals?.total_files || 0,
        totalSize: totals?.total_size || 0,
        avgFileSize: totals?.avg_file_size || 0,
        byMimeType: mimeTypeAggregates.map((row) => ({
          mimeType: row.mime_type,
          count: row.file_count,
          totalSize: row.sum_size,
          avgSize: row.avg_mime_size,
        })),
        largestFiles,
      };
    } catch (error) {
      statsLogger.error('Failed to get aggregate stats', {
        error: (error as Error).message,
      });
      throw error;
    }
  }

  // Helper to map database row (snake_case) to FileStats object (camelCase)
  private mapDbRowToFileStats(row: FileStatsDbRow): FileStats {
    return {
      id: row.id,
      fileName: row.file_name,
      filePath: row.file_path,
      mimeType: row.mime_type,
      size: row.size,
      lastModified: new Date(row.last_modified),
      width: row.width === null ? undefined : row.width,
      height: row.height === null ? undefined : row.height,
      duration: row.duration === null ? undefined : row.duration,
      bitrate: row.bitrate === null ? undefined : row.bitrate,
      encoding: row.encoding === null ? undefined : row.encoding,
      codec: row.codec === null ? undefined : row.codec,
      frameRate: row.frame_rate === null ? undefined : row.frame_rate,
      audioChannels: row.audio_channels === null ? undefined : row.audio_channels,
      sampleRate: row.sample_rate === null ? undefined : row.sample_rate,
      createdAt: new Date(row.created_at),
      updatedAt: new Date(row.updated_at),
    };
  }
}
