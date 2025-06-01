/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * src/modules/embeddings/embedding-http-client.ts
 * HTTP client for the CLIP embedding microservice (Refactored for Batch Processing)
 */

import axios, { AxiosError } from 'axios';
import fs from 'fs/promises'; // For getFileMetadata
import path from 'path'; // For getFileMetadata
import { execFile } from 'child_process'; // For getFileMetadata (ffprobe)
import { promisify } from 'util'; // For getFileMetadata
import imageSize from 'image-size'; // For getFileMetadata

import { config } from '../../config/server.config';
import { embeddingsLogger, EmbeddingComponent } from './embeddingsLogger';
import { ClipCache, ClipCacheEntry } from './embedding.service';
import { mapPythonMetadata } from './metadataUtils';
import Ajv, { ValidateFunction } from 'ajv';
import addFormats from 'ajv-formats';
import clipCacheSchema from '../../../schemas/clipCache.schema.json';

// Assuming FileHostingStatsHelper might be used by getFileMetadata if desired by the project context.
// If not, it can be removed along with its usage in getFileMetadata.
import { FileHostingStatsHelper } from '../file-hosting';
// getMimeType might be used by getFileMetadata or other parts, kept for now.
// import { getMimeType } from '../../utils/helpers'; // Not directly used in the provided snippets, consider removal if truly unused.

const execFileAsync = promisify(execFile);

// --- Constants ---
const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes default (for a batch HTTP request)
const HEALTH_CHECK_TIMEOUT_MS = 30 * 1000; // 30 seconds for health checks
const MIN_HTTP_TIMEOUT_MS = 10 * 1000; // 10 seconds minimum HTTP timeout for batch requests
const MAX_HTTP_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes maximum HTTP timeout for batch requests

export interface TimeoutConfig {
  baseTimeoutMs: number; // Base for batch HTTP timeout calculation
  healthCheckTimeoutMs: number;
  minTimeoutMs: number; // Minimum overall timeout for a batch call
  maxTimeoutMs: number; // Maximum overall timeout for a batch call
}

// --- Error Types ---
export class EmbeddingServiceError extends Error {
  statusCode?: number;
  responseData?: any;
  isNetworkError: boolean;
  isServiceError: boolean;
  originalError: Error;

  constructor(
    message: string,
    originalError: Error,
    options: {
      statusCode?: number;
      responseData?: any;
      isNetworkError?: boolean;
      isServiceError?: boolean;
    } = {}, // Added default empty object
  ) {
    super(message);
    this.name = 'EmbeddingServiceError';
    this.statusCode = options.statusCode;
    this.responseData = options.responseData;
    this.isNetworkError = options.isNetworkError || false;
    this.isServiceError = options.isServiceError || false;
    this.originalError = originalError;
  }
}

// --- Python Service Health Response ---
export interface EmbeddingServiceHealthResponse {
  status: 'ok' | string;
  uptime_seconds: number;
  processed_items_count: number;
  gpu_available: boolean;
  model_loaded: boolean;
  model_name?: string | null;
  device?: string | null;
  request_queue_size: number;
}

// --- Python Batch API Types ---
interface BatchMediaItemPayload {
  id: string;
  media_type: 'image' | 'video';
  source_type: 'filepath' | 'url';
  source: string;
  num_frames?: number;
  original_filename?: string;
}

interface PythonBatchRequest {
  items: BatchMediaItemPayload[];
  request_id?: string;
}

interface PythonEmbeddingResult {
  id: string;
  embedding: number[] | null;
  error: string | null;
  detail: string | null;
  debugMetadata?: any;
}

interface PythonBatchResponse {
  results: PythonEmbeddingResult[];
  batch_id: string;
  processed_by_request_id?: string;
}

// --- Input type for getEmbeddingsBatch ---
export interface MediaItemArgs {
  originalPath: string;
  resolvedPath: string;
  mediaType: 'image' | 'video';
  filename: string;
  fileSize: number;
  mtime: number;
  dimensions: { width: number; height: number };
  duration?: number | null;
  numFrames?: number | undefined; // Optional, if not provided, Python will handle it
}

// --- AJV Setup ---
const ajv = new Ajv({ allErrors: true });
addFormats(ajv);

let validateClipCacheEntry: ValidateFunction<ClipCacheEntry>;
try {
  if (!clipCacheSchema.definitions || !(clipCacheSchema.definitions as any).ClipCacheEntry) {
    throw new Error(
      'Schema definitions or ClipCacheEntry definition missing in clipCache.schema.json',
    );
  }
  validateClipCacheEntry = ajv.compile<ClipCacheEntry>(
    (clipCacheSchema.definitions as any).ClipCacheEntry,
  );
} catch (err: any) {
  embeddingsLogger.error(
    EmbeddingComponent.VALIDATION,
    'FATAL: Failed to compile ClipCacheEntry JSON Schema for EmbeddingHttpClient.',
    undefined,
    { error: String(err), details: err.stack },
  );
  validateClipCacheEntry = ((_data: any): _data is ClipCacheEntry => {
    const currentErrors = (validateClipCacheEntry as any).errors || [];
    (validateClipCacheEntry as any).errors = [
      ...currentErrors,
      { message: 'Schema compilation failed fatally.' },
    ];
    return false;
  }) as ValidateFunction<ClipCacheEntry>;
}

let fileStatsHelper: FileHostingStatsHelper | null = null;
async function getFileStatsHelperInstance(): Promise<FileHostingStatsHelper | null> {
  if (fileStatsHelper === null) {
    // Check if already attempted and failed
    try {
      const dbPath = path.join(process.cwd(), 'data', 'file_stats.db');
      const helper = new FileHostingStatsHelper(dbPath);
      await helper.initialize();
      fileStatsHelper = helper; // Assign only on success
    } catch (initError: any) {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        'Failed to initialize FileHostingStatsHelper in EmbeddingHttpClient. Stats-based metadata will be unavailable.',
        undefined,
        { error: String(initError), details: initError.stack },
      );
      fileStatsHelper = null; // Mark as attempted and failed to prevent retries
    }
  }
  return fileStatsHelper || null; // Return null if undefined (failed init)
}

export class EmbeddingHttpClient {
  private serviceUrl: string;
  private timeoutConfig: TimeoutConfig;

  constructor({
    serviceUrl = config.embedding.serviceUrl || 'http://localhost:3456',
    timeoutMs = DEFAULT_TIMEOUT_MS,
    timeoutConfig,
  }: {
    serviceUrl?: string;
    timeoutMs?: number;
    timeoutConfig?: Partial<TimeoutConfig>;
  } = {}) {
    this.serviceUrl = serviceUrl;
    this.timeoutConfig = {
      baseTimeoutMs: timeoutMs,
      healthCheckTimeoutMs: HEALTH_CHECK_TIMEOUT_MS,
      minTimeoutMs: MIN_HTTP_TIMEOUT_MS,
      maxTimeoutMs: MAX_HTTP_TIMEOUT_MS,
      ...timeoutConfig,
    };

    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      `Initialized EmbeddingHttpClient (Batch Mode).`,
      undefined,
      { serviceUrl: this.serviceUrl, timeoutConfig: this.timeoutConfig },
    );
  }

  public getServiceUrl(): string {
    return this.serviceUrl;
  }

  private categorizeError(error: any): EmbeddingServiceError {
    const networkErrorCodes = [
      'ECONNREFUSED',
      'ECONNRESET',
      'ETIMEDOUT',
      'ENOTFOUND',
      'ENETUNREACH',
      'EHOSTUNREACH',
      'ECONNABORTED',
    ];
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      let message = axiosError.message;
      let statusCode: number | undefined = undefined;
      let responseData: any = undefined;
      let isNetworkError = false;
      let isServiceError = false;

      if (axiosError.response) {
        isServiceError = true;
        statusCode = axiosError.response.status;
        responseData = axiosError.response.data;
        const detail =
          responseData?.detail ||
          responseData?.error ||
          (typeof responseData === 'string' ? responseData : '');
        message = `Embedding service responded with ${statusCode}${detail ? `: ${detail}` : ''}`;
      } else if (axiosError.request) {
        isNetworkError = true;
        message = `No response from embedding service: ${axiosError.message}`;
        if (axiosError.code && networkErrorCodes.includes(axiosError.code)) {
          message = `Network error (${axiosError.code}) connecting to embedding service: ${axiosError.message}`;
        }
        if (axiosError.code === 'ECONNABORTED' && message.toLowerCase().includes('timeout')) {
          message = `Request to embedding service timed out after ${axiosError.config?.timeout || 'configured'}ms.`;
        }
      } else {
        message = `Axios error setting up request to embedding service: ${axiosError.message}`;
      }
      return new EmbeddingServiceError(message, error, {
        statusCode,
        responseData,
        isNetworkError,
        isServiceError,
      });
    }
    return new EmbeddingServiceError(
      `Embedding service interaction failed: ${error.message}`,
      error,
      {},
    );
  }

  public async checkHealth(requestId: string): Promise<EmbeddingServiceHealthResponse> {
    embeddingsLogger.debug(
      EmbeddingComponent.SERVICE,
      'Checking embedding service health via HTTP client.',
      requestId,
    );
    try {
      const response = await axios.get<EmbeddingServiceHealthResponse>(
        `${this.serviceUrl}/health`,
        {
          timeout: this.timeoutConfig.healthCheckTimeoutMs,
          headers: { 'X-Request-ID': requestId },
        },
      );
      embeddingsLogger.info(
        EmbeddingComponent.SERVICE,
        `Health check successful: ${response.data.status}.`,
        requestId,
        { responseData: response.data },
      );
      return response.data;
    } catch (error: any) {
      const categorizedError = this.categorizeError(error);
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Health check failed: ${categorizedError.message}`,
        requestId,
        {
          error: categorizedError.message,
          isNetworkError: categorizedError.isNetworkError,
          statusCode: categorizedError.statusCode,
          details: categorizedError.originalError?.stack,
        },
      );
      throw categorizedError;
    }
  }

  public async getEmbeddingsBatch(
    mediaItems: MediaItemArgs[],
    requestId: string,
    clientTimeoutMs?: number,
  ): Promise<ClipCache> {
    const batchStartTime = Date.now();
    embeddingsLogger.updateContext(requestId, {
      mediaCount: mediaItems.length,
      operation: 'getEmbeddingsBatch',
    });

    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      `Processing ${mediaItems.length} media items as a single batch.`,
      requestId,
      { itemCount: mediaItems.length },
    );

    const pythonPayloadItems: BatchMediaItemPayload[] = mediaItems.map((item) => ({
      id: item.originalPath,
      media_type: item.mediaType,
      source_type: 'filepath', // Assuming 'filepath', adjust if URLs are directly supported via this flow
      source: item.resolvedPath,
      num_frames: item.numFrames,
      original_filename: item.filename,
    }));

    const pythonRequest: PythonBatchRequest = {
      items: pythonPayloadItems,
      request_id: requestId,
    };

    const perItemTimeoutIncrement = 1000; // 1 second per item (configurable if needed)
    const calculatedTimeout =
      (clientTimeoutMs || this.timeoutConfig.baseTimeoutMs) +
      mediaItems.length * perItemTimeoutIncrement;
    const finalHttpTimeout = Math.max(
      this.timeoutConfig.minTimeoutMs,
      Math.min(calculatedTimeout, this.timeoutConfig.maxTimeoutMs),
    );

    const clipCache: ClipCache = {};

    try {
      embeddingsLogger.debug(
        EmbeddingComponent.SERVICE,
        `Sending batch of ${mediaItems.length} items to Python service.`,
        requestId,
        {
          endpoint: `${this.serviceUrl}/api/embed_batch`,
          itemCount: mediaItems.length,
          timeout: finalHttpTimeout,
        },
      );

      const response = await axios.post<PythonBatchResponse>(
        `${this.serviceUrl}/api/embed_batch`,
        pythonRequest,
        {
          headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId },
          timeout: finalHttpTimeout,
        },
      );

      const pythonResults = response.data.results;
      const pythonBatchId = response.data.batch_id;
      embeddingsLogger.info(
        EmbeddingComponent.SERVICE,
        `Received ${pythonResults.length} results from Python for batch ${pythonBatchId}.`,
        requestId,
        {
          pythonBatchId,
          returnedItemCount: pythonResults.length,
          expectedItemCount: mediaItems.length,
        },
      );

      for (const pyResult of pythonResults) {
        const originalItem = mediaItems.find((m) => m.originalPath === pyResult.id);
        if (!originalItem) {
          embeddingsLogger.warn(
            EmbeddingComponent.VALIDATION,
            `Received result from Python for unknown item ID: ${pyResult.id}. Skipping.`,
            requestId,
            { unknownId: pyResult.id, pythonBatchId },
          );
          continue;
        }

        const mappedDebugMetadata = mapPythonMetadata(pyResult.debugMetadata);

        // Prefer the true Python-internal batch ID for embeddingConfig.batchId
        const resolvedBatchId =
          mappedDebugMetadata?.overallBatchRequestId ||
          mappedDebugMetadata?.batchId ||
          pythonBatchId ||
          'unknown';

        const cacheEntry: ClipCacheEntry = {
          schemaVersion: '1.1.0', // Standardized version
          filePath: originalItem.originalPath,
          mediaType: originalItem.mediaType,
          mtime: originalItem.mtime,
          fileSize: originalItem.fileSize,
          dimensions: originalItem.dimensions,
          duration: originalItem.duration || null,
          embedding: pyResult.embedding || [],
          embeddingModel: String(
            mappedDebugMetadata?.model || pyResult.debugMetadata?.model || 'unknown',
          ),
          embeddingConfig: {
            augmentation:
              typeof mappedDebugMetadata?.enableAugmentation === 'boolean'
                ? mappedDebugMetadata.enableAugmentation
                : undefined,
            numFrames:
              typeof mappedDebugMetadata?.numFrames === 'number'
                ? mappedDebugMetadata.numFrames
                : typeof mappedDebugMetadata?.num_extracted_frames === 'number'
                  ? mappedDebugMetadata.num_extracted_frames
                  : null,
            samplingMethod: String(
              mappedDebugMetadata?.methodUsed || mappedDebugMetadata?.sampling_method || 'unknown',
            ),
            deviceType: String(mappedDebugMetadata?.device || 'unknown'),
            batchId: String(resolvedBatchId),
          },
          processingTimestamp: String(
            mappedDebugMetadata?.processingTimestampUtc ||
              mappedDebugMetadata?.timestamp ||
              pyResult.debugMetadata?.timestamp ||
              new Date().toISOString(),
          ),
          debugMetadata: {
            ...(pyResult.debugMetadata || {}),
            ...mappedDebugMetadata,
            clientBatchHttpRequestTimeMs: Date.now() - batchStartTime,
            pythonBatchIdFromService: pythonBatchId,
            clientRequestId: requestId,
          },
          error: pyResult.error || undefined,
          detail: pyResult.detail || undefined,
        };

        if (!validateClipCacheEntry(cacheEntry)) {
          embeddingsLogger.warn(
            EmbeddingComponent.VALIDATION,
            `ClipCacheEntry validation failed for '${originalItem.originalPath}'. Attempting to fix.`,
            requestId,
            {
              filePath: originalItem.originalPath,
              validationErrors: JSON.stringify(validateClipCacheEntry.errors),
            },
          );
          this.fixCommonValidationIssues(cacheEntry);
          if (!validateClipCacheEntry(cacheEntry)) {
            embeddingsLogger.error(
              EmbeddingComponent.VALIDATION,
              `ClipCacheEntry still invalid after attempting fixes for '${originalItem.originalPath}'.`,
              requestId,
              {
                filePath: originalItem.originalPath,
                finalValidationErrors: JSON.stringify(validateClipCacheEntry.errors),
              },
            );
          }
        }
        clipCache[originalItem.originalPath] = cacheEntry;
      }

      if (pythonResults.length < mediaItems.length) {
        mediaItems.forEach((item) => {
          if (!clipCache[item.originalPath]) {
            const errorMessage = `Item '${item.originalPath}' was not included in Python's batch response for batch ${pythonBatchId}.`;
            embeddingsLogger.error(EmbeddingComponent.SERVICE, errorMessage, requestId, {
              originalPath: item.originalPath,
              pythonBatchId,
            });
            clipCache[item.originalPath] = this.createErrorClipCacheEntry(
              item.originalPath,
              new EmbeddingServiceError(errorMessage, new Error(errorMessage), {
                isServiceError: true,
              }),
              batchStartTime,
              item,
            );
          }
        });
      }
    } catch (error: any) {
      const categorizedError = this.categorizeError(error);
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `HTTP batch request to Python service failed: ${categorizedError.message}`,
        requestId,
        {
          error: categorizedError.message,
          details: categorizedError.originalError?.stack,
          isNetworkError: categorizedError.isNetworkError,
          isServiceError: categorizedError.isServiceError,
          statusCode: categorizedError.statusCode,
          responseData: categorizedError.responseData
            ? JSON.stringify(categorizedError.responseData).substring(0, 500)
            : undefined,
        },
      );
      for (const item of mediaItems) {
        clipCache[item.originalPath] = this.createErrorClipCacheEntry(
          item.originalPath,
          categorizedError,
          batchStartTime,
          item,
        );
      }
    }

    const totalClientTimeMs = Date.now() - batchStartTime;
    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      `Completed client-side processing for batch of ${mediaItems.length} items in ${totalClientTimeMs}ms.`,
      requestId,
      { itemCount: mediaItems.length, durationMs: totalClientTimeMs },
    );
    return clipCache;
  }

  private createErrorClipCacheEntry(
    originalPath: string,
    error: EmbeddingServiceError,
    batchStartTime: number,
    itemDetails: MediaItemArgs,
  ): ClipCacheEntry {
    const isTimeoutError = error.message.toLowerCase().includes('timeout');
    let errorMessage = `Error processing '${originalPath}' as part of a batch: ${error.message}`;
    if (isTimeoutError)
      errorMessage = `Timeout occurred while processing batch containing '${originalPath}'.`;

    const errorEntry: ClipCacheEntry = {
      schemaVersion: '1.1.0', // Standardized version
      filePath: originalPath,
      mediaType: itemDetails.mediaType,
      mtime: itemDetails.mtime,
      fileSize: itemDetails.fileSize,
      dimensions: itemDetails.dimensions,
      duration: itemDetails.duration || null,
      embedding: [],
      embeddingModel: 'unknown',
      embeddingConfig: {},
      processingTimestamp: new Date().toISOString(),
      error: errorMessage,
      detail: error.originalError?.stack || String(error.responseData) || error.message,
      debugMetadata: {
        errorType: isTimeoutError
          ? 'timeout_error'
          : error.isNetworkError
            ? 'network_error'
            : error.isServiceError
              ? 'service_error'
              : 'generic_processing_error',
        statusCode: error.statusCode,
        clientBatchProcessingTimeMsAtError: Date.now() - batchStartTime,
        originalErrorMessage: error.message,
      },
    };
    this.fixCommonValidationIssues(errorEntry);
    return errorEntry;
  }

  public async getFileMetadata(
    filePath: string, // Should be an absolute path resolvable by Node.js
    useStatsHelperOption = true, // Allow override if needed
  ): Promise<{
    mtime: number;
    fileSize: number;
    dimensions: { width: number; height: number };
    duration: number | null;
    mediaType: 'image' | 'video';
    filename: string;
  }> {
    const operationRequestId = `metadata-${path.basename(filePath)}-${Date.now()}`; // Unique ID for this specific operation
    embeddingsLogger.debug(
      EmbeddingComponent.SERVICE,
      `Fetching metadata for: ${filePath}`,
      operationRequestId,
      { filePath },
    );

    const statsHelperInstance = useStatsHelperOption ? await getFileStatsHelperInstance() : null;

    if (statsHelperInstance) {
      try {
        const mediaDir = config.mediaDir || path.join(process.cwd(), 'public', 'media');
        const relPath = path.isAbsolute(filePath) ? path.relative(mediaDir, filePath) : filePath;
        const fileStats = await statsHelperInstance.getStatsByPath(relPath);
        if (fileStats) {
          embeddingsLogger.debug(
            EmbeddingComponent.SERVICE,
            `Metadata from StatsHelper for ${filePath}`,
            operationRequestId,
            { fileStats },
          );
          return {
            mtime:
              fileStats.lastModified instanceof Date
                ? fileStats.lastModified.getTime()
                : new Date(fileStats.lastModified).getTime(),
            fileSize: fileStats.size,
            dimensions: { width: fileStats.width || 1, height: fileStats.height || 1 },
            duration: typeof fileStats.duration === 'number' ? fileStats.duration : null,
            mediaType: fileStats.mimeType.startsWith('video/') ? 'video' : 'image',
            filename: fileStats.fileName || path.basename(filePath),
          };
        }
      } catch (err: any) {
        embeddingsLogger.warn(
          EmbeddingComponent.SERVICE,
          `StatsHelper DB lookup failed for ${filePath}, falling back to direct FS. Error: ${err.message}`,
          operationRequestId,
          { error: err.stack },
        );
      }
    }

    try {
      const stats = await fs.stat(filePath);
      const ext = path.extname(filePath).toLowerCase();
      const mediaType = [
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
      let dimensions = { width: 1, height: 1 };
      let duration: number | null = null;

      if (mediaType === 'video') {
        try {
          const { stdout } = await execFileAsync('ffprobe', [
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=width,height,duration',
            '-of',
            'json',
            filePath,
          ]);
          const metadata = JSON.parse(stdout);
          if (metadata?.streams?.[0]) {
            dimensions = {
              width: parseInt(metadata.streams[0].width, 10) || 1,
              height: parseInt(metadata.streams[0].height, 10) || 1,
            };
            if (metadata.streams[0].duration && metadata.streams[0].duration !== 'N/A') {
              duration = parseFloat(metadata.streams[0].duration);
            }
          }
        } catch (ffprobeError: any) {
          embeddingsLogger.warn(
            EmbeddingComponent.SERVICE,
            `ffprobe failed for video ${filePath}: ${ffprobeError.message}`,
            operationRequestId,
            { error: ffprobeError.stderr || ffprobeError.stack },
          );
        }
      } else if (mediaType === 'image') {
        try {
          // image-size requires only a few starting bytes.
          const buffer = Buffer.alloc(4096); // Read enough bytes for most image formats
          const fd = await fs.open(filePath, 'r');
          const { bytesRead } = await fd.read(buffer, 0, buffer.length, 0);
          await fd.close();
          const imageInfo = imageSize(buffer.subarray(0, bytesRead)); // Use only bytes read
          if (imageInfo?.width && imageInfo?.height) {
            dimensions = { width: imageInfo.width, height: imageInfo.height };
          }
        } catch (imgErr: any) {
          embeddingsLogger.warn(
            EmbeddingComponent.SERVICE,
            `Failed to get image dimensions for ${filePath}: ${imgErr.message}`,
            operationRequestId,
            { error: imgErr.stack },
          );
        }
      }
      embeddingsLogger.debug(
        EmbeddingComponent.SERVICE,
        `Direct FS metadata for ${filePath}`,
        operationRequestId,
        { mtime: stats.mtimeMs, fileSize: stats.size, dimensions, duration, mediaType },
      );
      return {
        mtime: stats.mtimeMs,
        fileSize: stats.size,
        dimensions,
        duration,
        mediaType,
        filename: path.basename(filePath),
      };
    } catch (error: any) {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Critical metadata extraction failure for ${filePath}: ${error.message}`,
        operationRequestId,
        { error: error.stack },
      );
      throw new EmbeddingServiceError(`Metadata extraction failed for ${filePath}`, error, {});
    }
  }

  private fixCommonValidationIssues(entry: ClipCacheEntry): void {
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
}
