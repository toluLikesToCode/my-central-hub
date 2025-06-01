/**
 * src/modules/embeddings/embedding.service.ts
 * This file handles the embedding service logic for processing embeddings,
 * now by preparing a single batch request for the Python service.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

import path from 'path';
import fs from 'fs/promises'; // For findFile
import { execFile } from 'child_process'; // For ffprobe check
import { promisify } from 'util';

import Ajv, { ValidateFunction } from 'ajv';
import addFormats from 'ajv-formats';

import { embeddingsLogger, EmbeddingComponent } from './embeddingsLogger';
import { config } from '../../config/server.config';
import { EmbeddingHttpClient, MediaItemArgs, EmbeddingServiceError } from './embedding-http-client'; // Assuming MediaItemArgs is exported
import clipCacheSchema from '../../../schemas/clipCache.schema.json';

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
    'FATAL: Failed to compile ClipCacheEntry JSON Schema in EmbeddingService.',
    undefined,
    { error: String(err), details: err.stack },
  );
  validateEntry = ((_data: any): _data is ClipCacheEntry => {
    const currentErrors = (validateEntry as any).errors || [];
    (validateEntry as any).errors = [
      ...currentErrors,
      { message: 'Schema compilation failed fatally.' },
    ];
    return false;
  }) as ValidateFunction<ClipCacheEntry>;
}

// --- Configuration --- //
const SCRIPT_TIMEOUT_MS = config.embedding?.scriptTimeoutMs || 15 * 60 * 1000; // 15 minutes default for the entire batch HTTP call

// --- Status Types (Exported) --- //
export type EmbeddingServiceState = 'IDLE' | 'PROCESSING' | 'ERROR' | 'STOPPED'; // STARTING removed as Python service manages its own lifecycle.
export interface EmbeddingServiceStatus {
  state: EmbeddingServiceState;
  // pid is null as HTTP client doesn't track Python's PID.
  // isStarting is false as Python service manages its own lifecycle.
  // queueLength for Node.js service's internal queue is 0, Python has its own.
  pythonServiceHealth?: any; // Health details from Python service
  lastError?: string;
  isProcessingBatch: boolean; // Indicates if Node.js is currently awaiting a batch from Python
}

const execFileAsync = promisify(execFile);

/**
 * Manages the embedding service functionality by preparing batch requests
 * for an HTTP client that communicates with the CLIP embedding HTTP server.
 */
class EmbeddingService {
  private httpClient: EmbeddingHttpClient;
  private isProcessingBatch = false; // True if Node.js is currently awaiting a batch response from Python
  private lastError: string | null = null;

  constructor() {
    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      'Initializing Embedding Service (Node.js Intermediary for Python Batching).',
    );

    this.httpClient = new EmbeddingHttpClient({
      serviceUrl: config.embedding.serviceUrl, // Loaded from config by httpClient
      timeoutMs: SCRIPT_TIMEOUT_MS, // Default timeout for batch HTTP calls
      // httpClient manages its own retry logic if any
    });

    this.checkDependencies();
    this.setupExitHandlers();
  }

  private async checkDependencies() {
    try {
      await execFileAsync('ffprobe', ['-version']);
      embeddingsLogger.info(
        EmbeddingComponent.SERVICE,
        'Dependency check: ffprobe found for potential local metadata validation.',
      );
    } catch (error) {
      embeddingsLogger.warn(
        EmbeddingComponent.SERVICE,
        'Dependency check: ffprobe not found in PATH. Metadata relies entirely on Python service or limited local fs stats.',
        undefined,
        { error: (error as Error).message },
      );
    }
  }

  /**
   * Recursively searches inputDir (if set) and then publicDir for a file with the given filename.
   * Returns the absolute path if found, or null if not found.
   */
  private async findFile(filename: string, requestId?: string): Promise<string | null> {
    const searchDirs: { name: string; path?: string }[] = [
      { name: 'inputDir', path: config.embedding?.inputDir },
      { name: 'publicDir', path: config.publicDir || 'public' },
    ];

    for (const dirInfo of searchDirs) {
      if (dirInfo.path) {
        const absoluteDir = path.resolve(dirInfo.path);
        try {
          await fs.access(absoluteDir); // Check if dir exists and is accessible
          const foundInDir = await this._findFileInDirRecursive(filename, absoluteDir, requestId);
          if (foundInDir) {
            embeddingsLogger.debug(
              EmbeddingComponent.SERVICE,
              `File '${filename}' found in ${dirInfo.name} at '${foundInDir}'.`,
              requestId,
            );
            return foundInDir;
          }
        } catch (accessError) {
          embeddingsLogger.warn(
            EmbeddingComponent.SERVICE,
            `${dirInfo.name} '${absoluteDir}' not found or inaccessible, skipping.`,
            requestId,
            { directory: absoluteDir, error: (accessError as Error).message },
          );
        }
      }
    }
    embeddingsLogger.warn(
      EmbeddingComponent.SERVICE,
      `File '${filename}' not found in any configured search directories.`,
      requestId,
    );
    return null;
  }

  private async _findFileInDirRecursive(
    filename: string,
    dir: string,
    requestId?: string,
  ): Promise<string | null> {
    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch (err: any) {
      embeddingsLogger.warn(
        EmbeddingComponent.SERVICE,
        `Failed to read directory '${dir}': ${err.message}. Skipping.`,
        requestId,
        { directory: dir, error: err.message },
      );
      return null;
    }

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        const found = await this._findFileInDirRecursive(filename, fullPath, requestId);
        if (found) return found;
      } else if (entry.isFile() && entry.name === filename) {
        return fullPath;
      }
    }
    return null;
  }

  private _fixCommonValidationIssues(entry: ClipCacheEntry): void {
    const schemaProps = (clipCacheSchema.definitions as any).ClipCacheEntry.properties;

    if (typeof entry.schemaVersion !== 'string' || !entry.schemaVersion) {
      entry.schemaVersion =
        schemaProps.schemaVersion.const?.toString() || schemaProps.schemaVersion.default || '1.1.0';
    }
    if (typeof entry.filePath !== 'string' || !entry.filePath) entry.filePath = 'unknown_path';
    if (entry.mediaType !== 'image' && entry.mediaType !== 'video') {
      const ext = entry.filePath ? path.extname(entry.filePath).toLowerCase() : '';
      entry.mediaType = [
        '.mp4',
        '.mov',
        '.webm',
        '.avi',
        '.mkv',
        '.wmv',
        '.m4v',
        '.ogg',
        '.flv',
      ].includes(ext)
        ? 'video'
        : 'image';
    }
    if (typeof entry.mtime !== 'number' || isNaN(entry.mtime)) entry.mtime = 0;
    if (typeof entry.fileSize !== 'number' || isNaN(entry.fileSize)) entry.fileSize = 0;

    if (typeof entry.dimensions !== 'object' || entry.dimensions === null)
      entry.dimensions = { width: 1, height: 1 };
    if (
      typeof entry.dimensions.width !== 'number' ||
      isNaN(entry.dimensions.width) ||
      entry.dimensions.width <= 0
    )
      entry.dimensions.width = 1;
    if (
      typeof entry.dimensions.height !== 'number' ||
      isNaN(entry.dimensions.height) ||
      entry.dimensions.height <= 0
    )
      entry.dimensions.height = 1;

    if (entry.duration !== null && (typeof entry.duration !== 'number' || isNaN(entry.duration)))
      entry.duration = null;

    if (!Array.isArray(entry.embedding)) entry.embedding = [];
    entry.embedding = entry.embedding.filter((e) => typeof e === 'number' && !isNaN(e));

    if (typeof entry.embeddingModel !== 'string' || !entry.embeddingModel)
      entry.embeddingModel = 'unknown';
    if (typeof entry.embeddingConfig !== 'object' || entry.embeddingConfig === null)
      entry.embeddingConfig = {};

    if (
      typeof entry.processingTimestamp !== 'string' ||
      isNaN(new Date(entry.processingTimestamp).getTime())
    ) {
      entry.processingTimestamp = new Date().toISOString();
    }
    if (entry.error !== undefined && typeof entry.error !== 'string')
      entry.error = String(entry.error);
    if (entry.detail !== undefined && typeof entry.detail !== 'string')
      entry.detail = String(entry.detail);

    if (
      entry.debugMetadata !== undefined &&
      (typeof entry.debugMetadata !== 'object' || entry.debugMetadata === null)
    ) {
      entry.debugMetadata = { original_debug_metadata_was_invalid: entry.debugMetadata };
    }
  }

  private createLocalErrorClipCacheEntry(
    originalPath: string,
    errorMessage: string,
    errorDetail?: string,
    basicDetails?: Partial<
      Pick<ClipCacheEntry, 'mediaType' | 'mtime' | 'fileSize' | 'dimensions' | 'duration'>
    >,
  ): ClipCacheEntry {
    const now = new Date();
    const entry: ClipCacheEntry = {
      schemaVersion:
        (
          clipCacheSchema.definitions as any
        ).ClipCacheEntry.properties.schemaVersion.const?.toString() || '1.1.0',
      filePath: originalPath,
      mediaType: basicDetails?.mediaType || 'image', // Default, can be refined if ext is parsable
      mtime: basicDetails?.mtime || now.getTime(),
      fileSize: basicDetails?.fileSize || 0,
      dimensions: basicDetails?.dimensions || { width: 1, height: 1 },
      duration: basicDetails?.duration === undefined ? null : basicDetails.duration,
      embedding: [],
      embeddingModel: 'unknown',
      embeddingConfig: {},
      processingTimestamp: now.toISOString(),
      error: errorMessage,
      detail: errorDetail || 'Error originated in Node.js EmbeddingService.',
      debugMetadata: {
        nodeServiceError: true,
        reason: errorMessage,
      },
    };
    // Attempt to determine mediaType from extension if not provided
    if (!basicDetails?.mediaType && originalPath) {
      const ext = path.extname(originalPath).toLowerCase();
      if (['.mp4', '.mov', '.webm', '.avi', '.mkv', '.wmv', '.m4v', '.ogg', '.flv'].includes(ext)) {
        entry.mediaType = 'video';
      }
    }

    this._fixCommonValidationIssues(entry); // Apply fixes
    if (!validateEntry(entry)) {
      embeddingsLogger.warn(
        EmbeddingComponent.VALIDATION,
        `Local error ClipCacheEntry for '${originalPath}' failed validation even after fixes.`,
        undefined, // No specific request ID here, or pass if available
        {
          filePath: originalPath,
          validationErrors: JSON.stringify(validateEntry.errors),
          entryData: entry, // Log the problematic entry
        },
      );
    }
    return entry;
  }

  /**
   * Public method to request embeddings. It prepares a batch of media items
   * and sends a single request to the Python service via the HTTP client.
   */
  public async getEmbeddings(
    requestedImagePaths: string[], // Client-provided paths/IDs, used for finding files.
    requestId: string,
    rawPaths?: string[], // Exact original paths to be used as keys in the ClipCache.
    timeoutMs = SCRIPT_TIMEOUT_MS,
    numFrames?: number | null, // Optional frame count for videos, can be set by client.
  ): Promise<ClipCache> {
    const context = embeddingsLogger.createContext({
      requestId,
      mediaCount: requestedImagePaths.length,
      source: 'EmbeddingService.getEmbeddings',
    });

    // `originalPathsToUseAsKeys` are the keys that the client expects in the response.
    const originalPathsToUseAsKeys = rawPaths?.slice() ?? requestedImagePaths.slice();

    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      `Processing embedding request for ${originalPathsToUseAsKeys.length} paths.`,
      context,
    );
    this.lastError = null;

    const mediaItemsForPythonBatch: MediaItemArgs[] = [];
    // MEDIA_DIR on Node.js server, corresponding to PYTHON_MEDIA_ROOT in Python container
    const mediaDirAbsolute = path.resolve(config.mediaDir || 'public/media');

    for (const originalReqPath of originalPathsToUseAsKeys) {
      const filename = path.basename(originalReqPath);
      // `findFile` now uses `requestedImagePaths` elements (which are filenames or relative paths from client)
      // to locate absolute paths on the Node.js server.
      const foundAbsolutePath = await this.findFile(filename, requestId);

      if (foundAbsolutePath) {
        try {
          // Get metadata using the HTTP client's utility (which accesses local files)
          const metadata = await this.httpClient.getFileMetadata(foundAbsolutePath, true); // `true` to use FileHostingStatsHelper if available
          const relativePathForPython = path.relative(mediaDirAbsolute, foundAbsolutePath);

          mediaItemsForPythonBatch.push({
            originalPath: originalReqPath, // This is the key client expects.
            resolvedPath: relativePathForPython, // Path for Python, relative to its media root.
            mediaType: metadata.mediaType,
            filename: metadata.filename,
            fileSize: metadata.fileSize,
            mtime: metadata.mtime,
            dimensions: metadata.dimensions,
            duration: metadata.duration,
            numFrames: numFrames || undefined, // Use provided numFrames or null if not set
            // numFrames: Can be set here if specific frame counts per item are needed from Node.
            // Otherwise, Python service will use its default for videos.
          });
        } catch (metaError: any) {
          embeddingsLogger.warn(
            EmbeddingComponent.SERVICE,
            `Failed to get metadata for resolved file '${foundAbsolutePath}' (original key: '${originalReqPath}'). Skipping item for Python batch.`,
            context,
            { error: metaError.message, stack: metaError.stack },
          );
          // Error entry will be created later for this originalReqPath if it's not in Python's response.
        }
      } else {
        embeddingsLogger.warn(
          EmbeddingComponent.SERVICE,
          `File not found by Node.js server for original path key: '${originalReqPath}' (searched for filename: '${filename}'). It will be marked as error.`,
          context,
        );
        // Error entry will be created later.
      }
    }

    const finalCache: ClipCache = {};

    if (mediaItemsForPythonBatch.length === 0) {
      embeddingsLogger.warn(
        EmbeddingComponent.SERVICE,
        'No files were found or had metadata successfully extracted by Node.js. Returning errors for all requested paths.',
        context,
        { numRequested: originalPathsToUseAsKeys.length },
      );
      for (const reqPath of originalPathsToUseAsKeys) {
        finalCache[reqPath] = this.createLocalErrorClipCacheEntry(
          reqPath,
          'File not found or metadata extraction failed on Node.js server',
        );
      }
      embeddingsLogger.removeContext(requestId);
      return finalCache;
    }

    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      `Sending ${mediaItemsForPythonBatch.length} resolved items (out of ${originalPathsToUseAsKeys.length} requested) to Python service for batch embedding.`,
      context,
    );

    this.isProcessingBatch = true;
    try {
      const batchResultsCache = await this.httpClient.getEmbeddingsBatch(
        mediaItemsForPythonBatch,
        requestId,
        timeoutMs,
      );

      // Merge results from Python with local error creation for missing items
      for (const originalReqPath of originalPathsToUseAsKeys) {
        if (batchResultsCache[originalReqPath]) {
          finalCache[originalReqPath] = batchResultsCache[originalReqPath];
        } else {
          // This item was either not sent to Python (e.g., Node couldn't find it or metadata failed)
          // or Python did not return a result for it (httpClient should have created an error entry, but double check)
          embeddingsLogger.warn(
            EmbeddingComponent.SERVICE,
            `No result from Python for '${originalReqPath}'. Creating local error entry.`,
            context,
          );
          finalCache[originalReqPath] = this.createLocalErrorClipCacheEntry(
            originalReqPath,
            'File not processed: Not found by Node.js, metadata error, or missing from Python response.',
          );
        }
      }
      this.isProcessingBatch = false;
      embeddingsLogger.info(
        EmbeddingComponent.SERVICE,
        'Batch embedding request processed successfully by EmbeddingService.',
        context,
      );
      return finalCache;
    } catch (error: any) {
      this.lastError = error.message;
      this.isProcessingBatch = false;
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Node.js EmbeddingService: Batch request to Python service failed: ${error.message}`,
        context,
        {
          error: error instanceof EmbeddingServiceError ? error.message : String(error),
          isEmbeddingServiceError: error instanceof EmbeddingServiceError,
          originalErrorStack:
            error instanceof EmbeddingServiceError ? error.originalError?.stack : error.stack,
          details: error.details,
        },
      );

      // If the entire HTTP call failed, create error entries for all items *attempted* to be sent.
      for (const item of mediaItemsForPythonBatch) {
        // items that were prepared for Python
        finalCache[item.originalPath] = this.createLocalErrorClipCacheEntry(
          item.originalPath,
          `Batch processing failed: ${error.message}`,
          error.stack ||
            (error instanceof EmbeddingServiceError ? error.originalError?.message : String(error)),
          {
            mediaType: item.mediaType,
            fileSize: item.fileSize,
            mtime: item.mtime,
            dimensions: item.dimensions,
            duration: item.duration,
          },
        );
      }
      // Also ensure paths that were *not even prepared* get an error entry
      for (const originalReqPath of originalPathsToUseAsKeys) {
        if (!finalCache[originalReqPath]) {
          finalCache[originalReqPath] = this.createLocalErrorClipCacheEntry(
            originalReqPath,
            'File not processed due to prior batch failure or earlier error.',
            `Batch error: ${error.message}`,
          );
        }
      }
      // Do not throw the error if we want to return a ClipCache with error entries.
      // The current handler expects a ClipCache or throws, let's align by returning the cache.
      return finalCache;
    } finally {
      embeddingsLogger.removeContext(requestId);
    }
  }

  public async getStatus(requestId: string): Promise<EmbeddingServiceStatus> {
    const baseStatus = {
      isProcessingBatch: this.isProcessingBatch,
      lastError: this.lastError || undefined,
    };
    try {
      const pythonHealth = await this.httpClient.checkHealth(requestId);
      return {
        state:
          pythonHealth.status === 'ok' ? (this.isProcessingBatch ? 'PROCESSING' : 'IDLE') : 'ERROR',
        pythonServiceHealth: pythonHealth,
        ...baseStatus,
      };
    } catch (error: any) {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Failed to get Python service health: ${error.message}`,
        requestId,
        { error: error.stack },
      );
      return {
        state: 'ERROR',
        ...baseStatus,
        lastError:
          this.lastError || `Failed to connect/healthcheck Python service: ${error.message}`,
      };
    }
  }

  public stop(): void {
    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      'Node.js EmbeddingService stop requested. No active processes to stop locally. Python service manages its own lifecycle.',
    );
    // No specific action needed here as Python service is independent.
  }

  private setupExitHandlers() {
    const handleExit = (signalName: string) => {
      embeddingsLogger.info(
        EmbeddingComponent.SERVICE,
        `Node.js process exiting due to ${signalName}.`,
      );
      // Perform any cleanup if necessary in the future
    };

    process.on('exit', () => handleExit('exit'));
    process.on('SIGINT', () => {
      handleExit('SIGINT');
      process.exit(0);
    });
    process.on('SIGTERM', () => {
      handleExit('SIGTERM');
      process.exit(0);
    });
    process.on('uncaughtException', (err) => {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Uncaught Exception in Node.js EmbeddingService: ${err.message}`,
        undefined,
        { error: err, stack: err.stack },
      );
      // process.exit(1); // Consider if crashing is desired vs. attempting to continue
    });
    process.on('unhandledRejection', (reason, promise) => {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        'Unhandled Rejection in Node.js EmbeddingService.',
        undefined,
        { reason, promiseDetails: String(promise) },
      );
      // process.exit(1);
    });
  }
}

// Export a singleton instance
export const embeddingService = new EmbeddingService();
