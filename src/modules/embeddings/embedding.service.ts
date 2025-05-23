/**
 * src/modules/embeddings/embedding.service.ts
 * This file handles the embedding service logic for processing embeddings.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unused-vars */
import { execFile } from 'child_process';
import Ajv, { ValidateFunction } from 'ajv';
import addFormats from 'ajv-formats';
import clipCacheSchema from '../../../schemas/clipCache.schema.json';
import path from 'path';
import fs from 'fs/promises';
import { promisify } from 'util';
import { imageSize } from 'image-size';
import { embeddingsLogger, EmbeddingComponent } from './embeddingsLogger';
import { config } from '../../config/server.config';
import { randomUUID } from 'crypto';
import { EmbeddingHttpClient } from './embedding-http-client';

// --- Types --- //

/**
 * Represents the structure of a single entry in the ClipCache.
 */
export interface ClipCacheEntry {
  schemaVersion: string;
  filePath: string;
  mediaType: 'image' | 'video';
  mtime: number;
  fileSize: number;
  dimensions: { width: number; height: number };
  duration: number | null;
  embedding: number[];
  embeddingModel: string;
  embeddingConfig: {
    numFrames?: number | null;
    augmentation?: boolean;
    samplingMethod?: string;
    [k: string]: unknown;
  };
  processingTimestamp: string;
  debugMetadata?: { [k: string]: unknown };
  error?: string;
  detail?: string;
}

/**
 * Represents the entire ClipCache structure (filePath -> ClipCacheEntry mapping).
 */
export type ClipCache = Record<string, ClipCacheEntry>;

// --- AJV Setup --- //
const ajv = new Ajv({ allErrors: true });
addFormats(ajv);
// Compile the schema for a SINGLE entry - validation is crucial
let validateEntry: ValidateFunction<ClipCacheEntry>;
try {
  if (!clipCacheSchema.definitions || !(clipCacheSchema.definitions as any).ClipCacheEntry) {
    throw new Error(
      'Schema definitions or ClipCacheEntry definition missing in clipCache.schema.json',
    );
  }
  validateEntry = ajv.compile<ClipCacheEntry>((clipCacheSchema.definitions as any).ClipCacheEntry);
} catch (err: any) {
  embeddingsLogger.error(
    EmbeddingComponent.VALIDATION,
    'FATAL: Failed to compile ClipCacheEntry JSON Schema',
    undefined,
    { error: err },
  );

  // Using a dummy validator that always fails ensures no invalid data passes
  validateEntry = ((data: any) => {
    (validateEntry as any).errors = [{ message: 'Schema compilation failed' }];
    return false;
  }) as ValidateFunction<ClipCacheEntry>;
}

// --- Configuration --- //
const SCRIPT_TIMEOUT_MS = config.embedding?.scriptTimeoutMs || 15 * 60 * 1000; // 15 minutes default

// --- Types (Internal) --- //
interface FileMetadata {
  mtime: number;
  fileSize: number;
  dimensions: { width: number; height: number };
  duration: number | null;
  mediaType: 'image' | 'video';
}

// --- Status Types (Exported) --- //
export type EmbeddingServiceState = 'IDLE' | 'PROCESSING' | 'STARTING' | 'ERROR' | 'STOPPED';
export interface EmbeddingServiceStatus {
  state: EmbeddingServiceState;
  pid: number | null;
  isStarting: boolean;
  isProcessing: boolean;
  queueLength: number;
  currentBatch?: {
    count: number;
    total: number;
    current: string;
  };
  lastError?: string;
}

// Promisify execFile for ffprobe
const execFileAsync = promisify(execFile);

/**
 * Manages the embedding service functionality using an HTTP client
 * to communicate with the CLIP embedding HTTP server.
 */
class EmbeddingService {
  private httpClient: EmbeddingHttpClient;
  private isProcessing = false;
  private lastError: string | null = null;
  private currentBatch: { count: number; total: number; current: string } | null = null;

  constructor() {
    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      'Initializing Embedding Service with HTTP client...',
    );

    this.httpClient = new EmbeddingHttpClient({
      serviceUrl: config.embedding.serviceUrl || 'http://localhost:3456',
      maxRetries: config.embedding.maxRetries || 3,
      retryDelayMs: config.embedding.retryDelayMs || 1000,
      timeoutMs: config.embedding.timeoutMs || SCRIPT_TIMEOUT_MS,
    });

    this.checkDependencies();
    this.setupExitHandlers();
  }

  private checkDependencies() {
    // Check for ffprobe for metadata extraction
    try {
      execFileAsync('ffprobe', ['-version'])
        .then(() => {
          embeddingsLogger.info(EmbeddingComponent.SERVICE, 'Dependency check: ffprobe found.');
        })
        .catch((error) => {
          embeddingsLogger.error(
            EmbeddingComponent.SERVICE,
            'Dependency check failed: ffprobe not found in PATH. Video metadata extraction will fail.',
            undefined,
            { error },
          );
        });
    } catch (error) {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        'Dependency check failed: ffprobe not found in PATH. Video metadata extraction will fail.',
        undefined,
        { error },
      );
    }
  }

  // --- Metadata Fetching --- //

  /** Fetches metadata for a single file. */
  private async getFileMetadata(filePath: string): Promise<FileMetadata> {
    let mtime = 0;
    let fileSize = 0;
    let dimensions = { width: 1, height: 1 };
    let duration: number | null = null;
    let mediaType: 'image' | 'video' = 'image';

    try {
      const ext = path.extname(filePath).toLowerCase();
      // Basic media type detection based on extension
      mediaType = ['.mp4', '.mov', '.webm', '.avi', '.mkv', '.wmv', '.m4v'].includes(ext)
        ? 'video'
        : 'image';

      // 1. Get file stats (mtime, size)
      const stat = await fs.stat(filePath);
      mtime = stat.mtimeMs;
      fileSize = stat.size;

      // 2. Get dimensions (and duration for videos)
      if (mediaType === 'image') {
        try {
          // Optimization: Read only necessary bytes for image-size
          const buffer = Buffer.alloc(1024); // Adjust size if needed for specific formats
          const fd = await fs.open(filePath, 'r');
          await fd.read(buffer, 0, 1024, 0);
          await fd.close();
          const dim = imageSize(buffer);
          dimensions = { width: dim?.width ?? 1, height: dim?.height ?? 1 };
        } catch (imgErr) {
          embeddingsLogger.warn(
            EmbeddingComponent.SERVICE,
            `Failed to get image dimensions for ${filePath}: ${(imgErr as Error).message}. Using default 1x1.`,
            undefined,
            {
              filePath,
              error: imgErr,
            },
          );
          // Keep default dimensions
        }
      } else if (mediaType === 'video') {
        try {
          // Ensure ffprobe path is correct or in system PATH
          const { stdout } = await execFileAsync('ffprobe', [
            '-v',
            'error',
            '-select_streams',
            'v:0', // Select video stream 0
            '-show_entries',
            'stream=width,height,duration',
            '-of',
            'json', // Output as JSON
            filePath,
          ]);

          const info = JSON.parse(stdout);
          if (info.streams && info.streams[0]) {
            const s = info.streams[0];
            dimensions = { width: s.width ?? 1, height: s.height ?? 1 };
            duration = s.duration && !isNaN(parseFloat(s.duration)) ? parseFloat(s.duration) : null;
          } else {
            embeddingsLogger.warn(
              EmbeddingComponent.SERVICE,
              `ffprobe found no video stream info for ${filePath}.`,
              undefined,
              { filePath },
            );
          }
        } catch (ffprobeErr) {
          embeddingsLogger.warn(
            EmbeddingComponent.SERVICE,
            `ffprobe failed for ${filePath}: ${(ffprobeErr as Error).message}.`,
            undefined,
            {
              filePath,
              error: ffprobeErr,
            },
          );
          // Keep default dimensions/duration
        }
      }
    } catch (statErr) {
      embeddingsLogger.warn(
        EmbeddingComponent.SERVICE,
        `Failed to stat file ${filePath}: ${(statErr as Error).message}. Using default metadata.`,
        undefined,
        {
          filePath,
          error: statErr,
        },
      );
    }

    return { mtime, fileSize, dimensions, duration, mediaType };
  }

  /**
   * Recursively searches inputDir (if set) and then publicDir for a file with the given filename.
   * Returns the absolute path if found, or null if not found.
   */
  private async findFile(filename: string): Promise<string | null> {
    // 1. Check inputDir if configured and exists
    if (config.embedding?.inputDir) {
      const inputDir = path.resolve(config.embedding.inputDir);
      try {
        // verify the directory exists
        await fs.access(inputDir);
        const foundInInput = await this.findFileInDir(filename, inputDir);
        if (foundInInput) return foundInInput;
      } catch {
        // inputDir does not exist or is not readable
        embeddingsLogger.warn(
          EmbeddingComponent.SERVICE,
          `Input directory not found or inaccessible: ${inputDir}, skipping.`,
          undefined,
          { inputDir },
        );
      }
    }
    // 2. Check publicDir if configured, else default to 'public'
    const publicDir = path.resolve(config.publicDir || 'public');
    const foundInPublic = await this.findFileInDir(filename, publicDir);
    if (foundInPublic) return foundInPublic;
    return null;
  }

  /**
   * Helper: Recursively searches a directory for a file with the given filename.
   */
  private async findFileInDir(filename: string, dir: string): Promise<string | null> {
    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch (err: any) {
      embeddingsLogger.warn(
        EmbeddingComponent.SERVICE,
        `Failed to read directory ${dir}: ${err.message}. Skipping.`,
        undefined,
        {
          directory: dir,
          error: err,
        },
      );
      return null;
    }
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        const found = await this.findFileInDir(filename, fullPath);
        if (found) return found;
      } else if (entry.isFile() && entry.name === filename) {
        return fullPath;
      }
    }
    return null;
  }

  /**
   * Public method to request embeddings using the HTTP client.
   * Recursively searches the public directory for each filename in imagePaths.
   * The response will use the original request paths as keys and in filePath fields.
   */
  public async getEmbeddings(
    imagePaths: string[],
    requestId: string,
    rawPaths?: string[],
    timeoutMs = SCRIPT_TIMEOUT_MS,
  ): Promise<ClipCache> {
    const context = embeddingsLogger.createContext({
      requestId,
      mediaCount: imagePaths.length,
      source: 'getEmbeddings',
    });

    // originalPaths is the exact list we will use as JSON keys and filePath values
    const originalPaths = rawPaths?.slice() ?? imagePaths.slice();

    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      `Received embedding request for ${imagePaths.length} paths`,
      context,
      { pathsRequested: imagePaths.length },
    );

    // Reset error on new request
    this.lastError = null;

    // Map original request path to found absolute path
    const pathMap: Record<string, string | null> = {};
    for (const reqPath of originalPaths) {
      const filename = path.basename(reqPath);
      pathMap[reqPath] = await this.findFile(filename);
    }

    // Only use found absolute paths for processing
    const foundPaths = Object.values(pathMap).filter((p): p is string => !!p);
    if (foundPaths.length === 0) {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        'None of the requested files were found in the public directory.',
        context,
        { originalPaths },
      );

      embeddingsLogger.removeContext(requestId);
      return Promise.reject(
        new Error('None of the requested files were found in the public directory.'),
      );
    }

    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      `Found ${foundPaths.length}/${imagePaths.length} requested files`,
      context,
      {
        foundCount: foundPaths.length,
        requestedCount: imagePaths.length,
      },
    );

    try {
      // Set processing state
      this.isProcessing = true;
      this.currentBatch = {
        count: 0,
        total: foundPaths.length,
        current: foundPaths[0] || '',
      };

      // Use the HTTP client to get embeddings
      const clipCache = await this.httpClient.getEmbeddings(
        foundPaths,
        requestId,
        undefined,
        timeoutMs,
      );

      // Remap result: for each original request path, use the result for the found path
      const remapped: ClipCache = {};
      for (const reqPath of originalPaths) {
        const found = pathMap[reqPath];
        if (found && clipCache[found]) {
          remapped[reqPath] = { ...clipCache[found], filePath: reqPath };
        } else {
          embeddingsLogger.warn(
            EmbeddingComponent.SERVICE,
            `File not found in results: ${reqPath}`,
            context,
            { filePath: reqPath },
          );

          remapped[reqPath] = {
            schemaVersion: '1.1.0',
            filePath: reqPath,
            error: 'File not found in public directory',
            mtime: 0,
            fileSize: 0,
            dimensions: { width: 1, height: 1 },
            duration: null,
            mediaType: 'image',
            embedding: [],
            embeddingModel: 'unknown',
            embeddingConfig: {},
            processingTimestamp: new Date().toISOString(),
          };
        }
      }

      embeddingsLogger.info(EmbeddingComponent.SERVICE, `Request completed successfully`, context);

      this.isProcessing = false;
      this.currentBatch = null;
      embeddingsLogger.removeContext(requestId);

      return remapped;
    } catch (error: any) {
      this.lastError = error.message;
      this.isProcessing = false;
      this.currentBatch = null;

      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Request failed: ${error.message}`,
        context,
        { error },
      );

      embeddingsLogger.removeContext(requestId);
      throw error;
    }
  }

  // --- Service Control --- //

  /**
   * Get the current status of the embedding service.
   * Attempts to check the health of the HTTP embedding service.
   */
  public async getStatus(requestId: string): Promise<EmbeddingServiceStatus> {
    let state: EmbeddingServiceState = 'IDLE';
    let isHealthy = false;

    try {
      const healthResponse = await this.httpClient.checkHealth(requestId);
      isHealthy = healthResponse.status === 'ok';

      if (this.isProcessing) {
        state = 'PROCESSING';
      } else if (!isHealthy) {
        state = 'ERROR';
        this.lastError = 'Embedding service health check failed';

        // Log warning about service being unavailable
        embeddingsLogger.warn(
          EmbeddingComponent.SERVICE,
          'Embedding service health check failed - Python service may not be running',
          undefined,
          { serviceUrl: this.httpClient.getServiceUrl() },
        );
      } else {
        state = 'IDLE';
      }
    } catch (error: any) {
      state = 'ERROR';
      this.lastError = `Failed to connect to embedding service: ${error.message}`;
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Health check failed: ${error.message}`,
        undefined,
        { error },
      );

      // Log warning about service being unavailable
      embeddingsLogger.warn(
        EmbeddingComponent.SERVICE,
        'Cannot connect to embedding service - ensure Python service is running at the configured URL',
        undefined,
        { serviceUrl: this.httpClient.getServiceUrl() },
      );
    }

    const status = {
      state,
      pid: null, // HTTP client doesn't track PID
      isStarting: false,
      isProcessing: this.isProcessing,
      queueLength: 0, // HTTP client doesn't queue requests internally
      currentBatch: this.currentBatch ?? undefined,
      lastError: this.lastError ?? undefined,
    };

    embeddingsLogger.debug(
      EmbeddingComponent.SERVICE,
      `Status: ${status.state}`,
      undefined,
      status,
    );

    return status;
  }

  /**
   * Stop method (no-op for HTTP client, kept for API compatibility)
   */
  public stop(): void {
    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      'Stop requested, but HTTP client has no persistent processes to stop.',
    );
    return;
  }

  private setupExitHandlers() {
    // No active processes to kill, but keeping for consistency with original implementation
    const handleExit = () => {
      embeddingsLogger.info(EmbeddingComponent.SERVICE, 'Node process exiting.');
    };

    process.on('exit', handleExit);
    process.on('SIGINT', () => {
      embeddingsLogger.info(EmbeddingComponent.SERVICE, 'Received SIGINT.');
      handleExit();
      process.exit(0);
    });
    process.on('SIGTERM', () => {
      embeddingsLogger.info(EmbeddingComponent.SERVICE, 'Received SIGTERM.');
      handleExit();
      process.exit(0);
    });
    process.on('uncaughtException', (err) => {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Uncaught Exception: ${err.message}`,
        undefined,
        { error: err, stack: err.stack },
      );
      handleExit();
      process.exit(1);
    });
    process.on('unhandledRejection', (reason, promise) => {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Unhandled Rejection at: ${promise}, reason: ${reason}`,
        undefined,
        { reason, promise },
      );
      handleExit();
      process.exit(1);
    });
  }
}

// Export a singleton instance
export const embeddingService = new EmbeddingService();
