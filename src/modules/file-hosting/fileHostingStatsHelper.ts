/* eslint-disable @typescript-eslint/no-explicit-any */
import { open, Database } from 'sqlite';
import sqlite3 from 'sqlite3';
import { promises as fs } from 'fs';
import path from 'path';
import logger from '../../utils/logger';
import { getMimeType } from '../../utils/helpers';
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
  encoding?: string; // e.g., codec long name, pixel format
  codec?: string; // e.g., h264, mp3
  frameRate?: number;
  audioChannels?: number;
  sampleRate?: number;
  createdAt: Date;
  updatedAt: Date;
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

interface FFProbeStream {
  index: number;
  codec_name?: string;
  codec_long_name?: string;
  codec_type?: 'video' | 'audio' | 'subtitle' | 'data' | 'attachment';
  width?: number;
  height?: number;
  duration?: string; // Can be string
  bit_rate?: string; // Can be string
  r_frame_rate?: string; // e.g., "25/1"
  channels?: number;
  channel_layout?: string;
  sample_rate?: string; // Can be string
  pix_fmt?: string;
  // ... other stream properties
}

interface FFProbeFormat {
  filename: string;
  nb_streams: number;
  nb_programs: number;
  format_name: string;
  format_long_name: string;
  start_time?: string;
  duration?: string; // Can be string
  size?: string; // Can be string
  bit_rate?: string; // Can be string
  probe_score?: number;
  // ... other format_tags
  tags?: Record<string, string>;
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
      const dir = path.dirname(this.dbPath);
      await fs.mkdir(dir, { recursive: true });

      this.db = await open({
        filename: this.dbPath,
        driver: sqlite3.Database,
      });

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

  private async getMediaInfo(filePath: string): Promise<FFProbeOutput | null> {
    try {
      const { stdout } = await execFileAsync('ffprobe', [
        '-v',
        'error',
        '-show_format',
        '-show_streams',
        '-print_format',
        'json',
        filePath,
      ]);
      return JSON.parse(stdout) as FFProbeOutput;
    } catch (error) {
      statsLogger.warn(`Failed to execute ffprobe for ${filePath}`, {
        error: (error as Error).message,
      });
      return null;
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

      const fileStats: FileStats = {
        fileName,
        filePath: relativePath,
        mimeType,
        size: stats.size,
        lastModified: stats.mtime,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      const mediaInfo = await this.getMediaInfo(filePath);

      if (mediaInfo) {
        if (mimeType.startsWith('image/')) {
          this.populateImageStats(mediaInfo, fileStats);
        } else if (mimeType.startsWith('video/')) {
          this.populateVideoStats(mediaInfo, fileStats);
        } else if (mimeType.startsWith('audio/')) {
          this.populateAudioStats(mediaInfo, fileStats);
        }

        // Common properties from format if not already set or to override
        if (mediaInfo.format) {
          if (fileStats.duration === undefined) {
            fileStats.duration = safeParseFloat(mediaInfo.format.duration);
          }
          // Overall bitrate is often preferred
          const formatBitrate = safeParseInt(mediaInfo.format.bit_rate);
          if (formatBitrate !== undefined) {
            fileStats.bitrate = formatBitrate;
          }
        }
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
      return !!(result && result.count > 0);
    } catch (error) {
      statsLogger.error(`Failed to check stats for file: ${filePath}`, {
        error: (error as Error).message,
      });
      return false;
    }
  }

  private populateImageStats(mediaInfo: FFProbeOutput, fileStats: FileStats): void {
    const imageStream = mediaInfo.streams.find((s) => s.codec_type === 'video');
    if (imageStream) {
      fileStats.width = safeParseInt(imageStream.width);
      fileStats.height = safeParseInt(imageStream.height);
      fileStats.codec = imageStream.codec_name;
      fileStats.encoding = imageStream.codec_long_name || imageStream.pix_fmt; // Use codec_long_name or pix_fmt for encoding
      // Bitrate for single images is often not applicable or not provided by ffprobe stream
      // It might be in format.bit_rate if the image is in a container format
      const streamBitrate = safeParseInt(imageStream.bit_rate);
      if (fileStats.bitrate === undefined && streamBitrate !== undefined) {
        fileStats.bitrate = streamBitrate;
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
      fileStats.encoding = videoStream.codec_long_name || videoStream.pix_fmt;
      fileStats.duration = safeParseFloat(videoStream.duration); // Stream duration
      fileStats.frameRate = parseFrameRate(videoStream.r_frame_rate);

      const videoBitrate = safeParseInt(videoStream.bit_rate);
      if (videoBitrate !== undefined) {
        fileStats.bitrate = videoBitrate; // Prefer video stream bitrate for video files if format.bit_rate is not specific enough
      }
    }

    if (audioStream) {
      fileStats.audioChannels = safeParseInt(audioStream.channels);
      fileStats.sampleRate = safeParseInt(audioStream.sample_rate);
      // codec, encoding, bitrate on FileStats are typically for the primary (video) stream.
      // If separate audio codec/bitrate is needed, FileStats interface should be extended.
    }

    // Prefer format duration and bitrate if available and more general
    if (mediaInfo.format) {
      const formatDuration = safeParseFloat(mediaInfo.format.duration);
      if (formatDuration !== undefined) fileStats.duration = formatDuration;

      const formatBitrate = safeParseInt(mediaInfo.format.bit_rate);
      if (formatBitrate !== undefined) fileStats.bitrate = formatBitrate; // Override with overall bitrate
    }
  }

  private populateAudioStats(mediaInfo: FFProbeOutput, fileStats: FileStats): void {
    const audioStream = mediaInfo.streams.find((s) => s.codec_type === 'audio');
    if (audioStream) {
      fileStats.codec = audioStream.codec_name;
      fileStats.encoding = audioStream.codec_long_name;
      fileStats.audioChannels = safeParseInt(audioStream.channels);
      fileStats.sampleRate = safeParseInt(audioStream.sample_rate);
      fileStats.duration = safeParseFloat(audioStream.duration);
      fileStats.bitrate = safeParseInt(audioStream.bit_rate);
    }

    // Prefer format duration and bitrate if available and more general
    if (mediaInfo.format) {
      const formatDuration = safeParseFloat(mediaInfo.format.duration);
      if (formatDuration !== undefined) fileStats.duration = formatDuration;

      const formatBitrate = safeParseInt(mediaInfo.format.bit_rate);
      if (formatBitrate !== undefined) fileStats.bitrate = formatBitrate; // Override with overall bitrate
    }
  }

  // getImageStats, getVideoStats, getAudioStats are replaced by the new populate methods
  // and the generic getMediaInfo call in getFileStats.

  /**
   * Store file statistics in the database
   * @param fileStats File statistics to store
   */
  async saveFileStats(fileStats: FileStats): Promise<number> {
    if (!this.initialized) await this.initialize();
    if (!this.db) throw new Error('Database not initialized');

    try {
      const now = new Date(); // Use current date for updatedAt

      // Ensure createdAt is set, if not (e.g. direct call to saveFileStats), set it.
      const createdAt = fileStats.createdAt instanceof Date ? fileStats.createdAt : new Date();

      const result = await this.db.run(
        `
        INSERT INTO file_stats (
          file_name, file_path, mime_type, size, last_modified,
          width, height, duration, bitrate, encoding,
          codec, frame_rate, audio_channels, sample_rate,
          created_at, updated_at
        ) VALUES (
          ?, ?, ?, ?, ?, 
          ?, ?, ?, ?, ?, 
          ?, ?, ?, ?, ?, 
          ?
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
          -- created_at is NOT updated on conflict, keep original
          updated_at = excluded.updated_at 
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
          createdAt.toISOString(), // For new inserts
          now.toISOString(), // For new inserts and updates
        ],
      );

      statsLogger.debug(`Stats saved for file: ${fileStats.fileName}`, {
        path: fileStats.filePath,
        size: fileStats.size,
        id: result.lastID, // lastID is for INSERTs. For UPSERTs, changes indicates success.
        changes: result.changes,
      });

      // If lastID is 0 but changes were made, it was an update. We might need to fetch the ID.
      // For simplicity, returning lastID (for inserts) or a placeholder if it was an update.
      // The prompt does not specify behavior for return value on update.
      // If ID is crucial on update, a SELECT would be needed post-UPSERT.
      return (
        result.lastID ||
        (result.changes ? (await this.getStatsByPath(fileStats.filePath))?.id || 0 : 0)
      );
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
      const row = await this.db.get<FileStatsDbRow>( // Use a type for DB row
        `
        SELECT * FROM file_stats WHERE file_path = ?
      `,
        [filePath],
      );

      if (!row) return null;

      return this.mapDbRowToFileStats(row);
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
      sortBy?: keyof FileStats; // Allow sorting
      sortOrder?: 'ASC' | 'DESC';
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
      // ... (other conditions from original code are fine)
      if (options.minSize !== undefined) {
        conditions.push('size >= ?');
        params.push(options.minSize);
      }
      if (options.maxSize !== undefined) {
        conditions.push('size <= ?');
        params.push(options.maxSize);
      }
      if (options.hasAudio) {
        conditions.push('audio_channels IS NOT NULL AND audio_channels > 0');
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

      const validSortColumns: (keyof FileStats)[] = [
        // Whitelist sortable columns
        'fileName',
        'mimeType',
        'size',
        'lastModified',
        'width',
        'height',
        'duration',
        'bitrate',
        'createdAt',
        'updatedAt',
      ];
      const sortBy =
        options.sortBy && validSortColumns.includes(options.sortBy)
          ? options.sortBy
          : 'last_modified';
      // Map FileStats key to DB column name if different (e.g. fileName -> file_name)
      const dbSortBy = sortBy === 'lastModified' ? 'last_modified' : sortBy; // Add other mappings if needed
      const sortOrder = options.sortOrder === 'ASC' ? 'ASC' : 'DESC';
      const orderByClause = `ORDER BY ${dbSortBy} ${sortOrder}`;

      const limitClause = options.limit ? `LIMIT ${safeParseInt(options.limit) || 10}` : ''; // Default limit if needed
      const offsetClause = options.offset ? `OFFSET ${safeParseInt(options.offset) || 0}` : '';

      const rows = await this.db.all<FileStatsDbRow[]>( // Use a type for DB row
        `
        SELECT * FROM file_stats
        ${whereClause}
        ${orderByClause}
        ${limitClause}
        ${offsetClause}
      `,
        params,
      );

      return rows.map(this.mapDbRowToFileStats);
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
    largestFiles: FileStats[]; // This should be sorted by size by default
  }> {
    if (!this.initialized) await this.initialize();
    if (!this.db) throw new Error('Database not initialized');

    try {
      const totals = await this.db.get<{
        total_files: number;
        total_size: number;
        avg_size: number;
      }>(`
        SELECT COUNT(*) as total_files, SUM(size) as total_size, AVG(size) as avg_size
        FROM file_stats
      `);

      const byMimeType = await this.db.all<
        { mime_type: string; count: number; total_size: number }[]
      >(`
        SELECT 
          mime_type,
          COUNT(*) as count,
          SUM(size) as total_size
        FROM file_stats
        GROUP BY mime_type
        ORDER BY total_size DESC
      `);

      // Get largest files by querying with sort options
      const largestFilesRaw = await this.queryFileStats({
        sortBy: 'size',
        sortOrder: 'DESC',
        limit: 10,
      });

      return {
        totalFiles: totals?.total_files || 0,
        totalSize: totals?.total_size || 0,
        avgFileSize: totals?.avg_size || 0,
        byMimeType: byMimeType.map((row) => ({
          mimeType: row.mime_type,
          count: row.count,
          totalSize: row.total_size,
        })),
        largestFiles: largestFilesRaw,
      };
    } catch (error) {
      statsLogger.error('Failed to get aggregate stats', {
        error: (error as Error).message,
      });
      throw error;
    }
  }

  // Helper to map database row to FileStats object
  private mapDbRowToFileStats(row: FileStatsDbRow): FileStats {
    return {
      id: row.id,
      fileName: row.file_name,
      filePath: row.file_path,
      mimeType: row.mime_type,
      size: row.size,
      lastModified: new Date(row.last_modified),
      width: row.width === null ? undefined : row.width, // handle null from DB
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

// Define a type for the database row structure for better type safety
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
