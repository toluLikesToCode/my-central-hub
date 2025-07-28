import path from 'path';
import { FileHostingService, FileInfo } from './fileHostingService';
import { FileHostingStatsHelper } from './fileHostingStatsHelper';
import { config } from '../../config/server.config';
import logger from '../../utils/logger';

const statsLogger = logger.child({
  module: 'file-hosting',
  component: 'stats-initializer',
});

/**
 * Class to initialize stats for files without existing stats
 * Runs once at application startup
 */
export class FileStatsInitializer {
  private fileService: FileHostingService;
  private statsHelper: FileHostingStatsHelper;
  private staticDir: string;

  constructor() {
    this.staticDir = config.staticDir;
    this.fileService = new FileHostingService(this.staticDir);
    this.statsHelper = new FileHostingStatsHelper(
      path.join(process.cwd(), 'data', 'file_stats.db'),
    );
  }

  /**
   * Initialize the stats database and update missing file stats
   */
  public async initialize(): Promise<void> {
    statsLogger.info('Starting file stats initialization at startup');
    const startTime = Date.now();

    try {
      // Initialize the database
      await this.statsHelper.initialize();

      // Get all files recursively
      const allFiles = await this.fileService.listFiles('.', true);
      const fileInfos: FileInfo[] = [];

      // Convert mixed array to FileInfo array
      if (Array.isArray(allFiles)) {
        for (const item of allFiles) {
          if (typeof item !== 'string' && !item.isDirectory) {
            fileInfos.push(item);
          }
        }
      }

      statsLogger.info(`Found ${fileInfos.length} files total in hosting directory`);

      // Get list of files that don't have stats yet
      const filesWithoutStats = await this.filterFilesWithoutStats(fileInfos);

      if (filesWithoutStats.length === 0) {
        statsLogger.info('All files already have stats. No updates needed.');
        return;
      }

      statsLogger.info(
        `Found ${filesWithoutStats.length} files without stats that need processing`,
      );

      // Process files in batches
      const batchSize = 10;
      let processed = 0;
      let failures = 0;

      for (let i = 0; i < filesWithoutStats.length; i += batchSize) {
        const batch = filesWithoutStats.slice(i, i + batchSize);

        // Process each file in the batch concurrently
        const promises = batch.map(async (file) => {
          try {
            if (!file.path) return;

            const filePath = file.path.toString();
            const absolutePath = path.join(this.staticDir, filePath);

            // Get detailed file statistics
            const fileStats = await this.statsHelper.getFileStats(absolutePath, this.staticDir);

            // Save to database
            await this.statsHelper.saveFileStats(fileStats);

            processed++;
          } catch (error) {
            failures++;
            statsLogger.error(`Failed to process file: ${file.path}`, {
              error: (error as Error).message,
            });
          }
        });

        // Wait for the current batch to complete
        await Promise.all(promises);
      }

      const duration = (Date.now() - startTime) / 1000;
      statsLogger.info(`File stats initialization completed`, {
        totalFilesChecked: fileInfos.length,
        filesNeedingStats: filesWithoutStats.length,
        processed,
        failures,
        duration: `${duration.toFixed(2)} seconds`,
      });
    } catch (error) {
      statsLogger.error('Failed to initialize file statistics', {
        error: (error as Error).message,
        stack: (error as Error).stack,
      });
      // Don't rethrow to prevent app startup failure
    }
  }

  /**
   * Filter files to find which ones don't have stats yet
   */
  private async filterFilesWithoutStats(files: string[] | FileInfo[]): Promise<FileInfo[]> {
    const result: FileInfo[] = [];

    for (const file of files) {
      // Skip if it's a string or doesn't have a path property
      if (typeof file === 'string' || !('path' in file) || !file.path) continue;

      const filePath = file.path.toString();

      // Check if stats exist for this file
      const hasStats = await this.statsHelper.checkStatsExist(filePath);

      if (!hasStats) {
        result.push(file);
      }
    }

    return result;
  }

  /**
   * Close resources when done
   */
  public async close(): Promise<void> {
    await this.statsHelper.close();
  }
}

/**
 * Function to run at application startup
 */
export async function initializeFileStats(): Promise<void> {
  const initializer = new FileStatsInitializer();
  try {
    await initializer.initialize();
  } finally {
    await initializer.close();
  }
}
