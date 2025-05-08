/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * src/modules/embeddings/embedding-http-client.ts
 * HTTP client for the CLIP embedding microservice
 */

import axios, { AxiosError } from 'axios';
import fs from 'fs/promises';
import path from 'path';
import { config } from '../../config/server.config';
import { embeddingsLogger, EmbeddingComponent } from './embeddingsLogger';
import { ClipCache, ClipCacheEntry } from './embedding.service';
import { mapPythonMetadata } from './metadataUtils';
import Ajv, { ValidateFunction } from 'ajv';
import addFormats from 'ajv-formats';
import clipCacheSchema from '../../../schemas/clipCache.schema.json'; // Correct path to top-level schemas folder
import { execFile } from 'child_process';
import { promisify } from 'util';
import imageSize from 'image-size';
import { createReadStream } from 'fs'; // Use streams for raw byte upload
const execFileAsync = promisify(execFile);

// Default retry configuration
const DEFAULT_RETRY_COUNT = 3;
const DEFAULT_RETRY_DELAY_MS = 1000;

// Timeout configuration
const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes default
const HEALTH_CHECK_TIMEOUT_MS = 5 * 1000; // 10 seconds for health checks
const MIN_TIMEOUT_MS = 30 * 1000; // 30 seconds minimum timeout
const MAX_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes maximum timeout
const VIDEO_TIMEOUT_MULTIPLIER = 4; // Videos get 4x the timeout of images
const MB_SIZE = 1024 * 1024; // 1 MB in bytes
const TIMEOUT_PER_MB = 1000; // 1 second per MB of file size

// Define a configuration interface for timeout settings
export interface TimeoutConfig {
  baseTimeoutMs: number; // Base timeout for all requests
  healthCheckTimeoutMs: number; // Timeout for health check requests
  minTimeoutMs: number; // Minimum timeout regardless of file size
  maxTimeoutMs: number; // Maximum timeout cap
  videoMultiplier: number; // Multiplier for video files
  timeoutPerMbMs: number; // Additional timeout per MB of file size
}

/**
 * Custom error types for better error handling
 */
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
    },
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

/**
 * Response from the /health endpoint
 */
export interface EmbeddingServiceHealthResponse {
  status: 'ok' | string;
  uptime_seconds: number;
  processed_files: number;
  gpu_available: boolean;
  model_loaded: boolean;
  model_name: string;
  device: string;
}

/**
 * Response from the /embed endpoint with detailed typings for debugMetadata
 */
export interface EmbeddingResponse {
  embedding: number[];
  debugMetadata?: {
    // Core metadata fields
    model: string;
    enable_augmentation: boolean;
    timestamp: string;
    request_id: string;
    processing_time_ms: number;

    // Device information
    device?: string; // e.g., 'cuda', 'mps', 'cpu'

    // Batch processing metadata
    batch_id?: string;

    // Video processing metadata
    num_frames?: number;
    method_used?: 'scene_detection' | 'fallback_uniform' | string;
    scene_count?: number;
    selected_times?: number[];
    entropy_values?: Array<[number, number]>; // [timestamp, entropy_value]
    selected_entropy_values?: Array<[number, number]>;

    // Additional metadata fields that might be present
    [key: string]: any;
  };
  error: string | null;
  detail: string | null;
}

// Initialize AJV validator for ClipCacheEntry
const ajv = new Ajv({ allErrors: true });
addFormats(ajv);

// Compile validator for ClipCacheEntry using the schema
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
    'FATAL: Failed to compile ClipCacheEntry JSON Schema',
    undefined,
    { error: err },
  );

  // Create a dummy validator that always returns true if schema compilation fails
  // This is a fallback to allow the system to function even with schema issues
  validateClipCacheEntry = (() => true) as unknown as ValidateFunction<ClipCacheEntry>;
}

/**
 * HTTP client for communicating with the CLIP embedding microservice
 */
export class EmbeddingHttpClient {
  private serviceUrl: string;
  private maxRetries: number;
  private retryDelayMs: number;
  private baseTimeoutMs: number;
  private timeoutConfig: TimeoutConfig;

  constructor({
    serviceUrl = config.embedding.serviceUrl || 'http://localhost:3456',
    maxRetries = DEFAULT_RETRY_COUNT,
    retryDelayMs = DEFAULT_RETRY_DELAY_MS,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    timeoutConfig,
  }: {
    serviceUrl?: string;
    maxRetries?: number;
    retryDelayMs?: number;
    timeoutMs?: number;
    timeoutConfig?: Partial<TimeoutConfig>;
  } = {}) {
    this.serviceUrl = serviceUrl;
    this.maxRetries = maxRetries;
    this.retryDelayMs = retryDelayMs;
    this.baseTimeoutMs = timeoutMs;

    // Initialize timeout configuration with defaults and any overrides
    this.timeoutConfig = {
      baseTimeoutMs: timeoutMs,
      healthCheckTimeoutMs: HEALTH_CHECK_TIMEOUT_MS,
      minTimeoutMs: MIN_TIMEOUT_MS,
      maxTimeoutMs: MAX_TIMEOUT_MS,
      videoMultiplier: VIDEO_TIMEOUT_MULTIPLIER,
      timeoutPerMbMs: TIMEOUT_PER_MB,
      ...timeoutConfig,
    };

    // Log initialization with timeout configuration
    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      `Initialized HTTP embedding client with serviceUrl: ${this.serviceUrl}, baseTimeout: ${this.baseTimeoutMs}ms`,
      undefined,
      {
        serviceUrl: this.serviceUrl,
        timeoutConfig: this.timeoutConfig,
      },
    );
  }

  /**
   * Get the service URL for diagnostics
   */
  public getServiceUrl(): string {
    return this.serviceUrl;
  }

  /**
   * Calculate an appropriate timeout based on file metadata
   * @param metadata File metadata including size and media type
   * @returns Calculated timeout in milliseconds
   */
  private calculateTimeout(metadata: {
    fileSize: number;
    mediaType: 'image' | 'video';
    duration?: number | null;
  }): number {
    // Simplified timeout calculation with optional video duration
    let timeout = this.timeoutConfig.baseTimeoutMs;
    const fileSizeMb = metadata.fileSize / MB_SIZE;
    timeout += Math.max(0, fileSizeMb * this.timeoutConfig.timeoutPerMbMs);
    if (metadata.mediaType === 'video') {
      const durationMultiplier = metadata.duration ? Math.max(1, metadata.duration / 60) : 1; // 1x per minute
      timeout = timeout * this.timeoutConfig.videoMultiplier * durationMultiplier;
    }
    // Enforce minimum and maximum bounds
    timeout = Math.max(timeout, this.timeoutConfig.minTimeoutMs);
    timeout = Math.min(timeout, this.timeoutConfig.maxTimeoutMs);
    return timeout;
  }

  /**
   * Categorize an error as network or service error
   * @param error Axios error or any other error
   * @returns Categorized error with additional context
   */
  private categorizeError(error: any): EmbeddingServiceError {
    // Network errors (connection refused, timeout, etc.)
    const networkErrorCodes = [
      'ECONNREFUSED',
      'ECONNRESET',
      'ETIMEDOUT',
      'ENOTFOUND',
      'ENETUNREACH',
      'EHOSTUNREACH',
    ];

    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;

      // Case 1: Network error (no response received)
      if (!axiosError.response) {
        const isKnownNetworkError = axiosError.code && networkErrorCodes.includes(axiosError.code);
        const errorMessage = isKnownNetworkError
          ? `Network error connecting to embedding service (${axiosError.code}): ${axiosError.message}`
          : `Network error connecting to embedding service: ${axiosError.message}`;

        return new EmbeddingServiceError(errorMessage, error, {
          isNetworkError: true,
          isServiceError: false,
        });
      }

      // Case 2: Service responded with an error status code
      const statusCode = axiosError.response.status;
      const responseData = axiosError.response.data;

      // Different error messages based on status code ranges
      let errorMessage = '';
      if (statusCode >= 500) {
        errorMessage = `Embedding service internal error (${statusCode}): ${axiosError.message}`;
      } else if (statusCode >= 400) {
        errorMessage = `Embedding service request error (${statusCode}): ${axiosError.message}`;
      } else {
        errorMessage = `Embedding service error (${statusCode}): ${axiosError.message}`;
      }

      return new EmbeddingServiceError(errorMessage, error, {
        statusCode,
        responseData,
        isNetworkError: false,
        isServiceError: true,
      });
    }

    // Case 3: Generic error (not Axios-specific)
    return new EmbeddingServiceError(`Embedding service error: ${error.message}`, error, {
      isNetworkError: false,
      isServiceError: false,
    });
  }

  /**
   * Check the health of the embedding service
   */
  public async checkHealth(requestId: string): Promise<EmbeddingServiceHealthResponse> {
    const context = embeddingsLogger.createContext({
      requestId,
      source: 'checkHealth',
    });

    embeddingsLogger.debug(
      EmbeddingComponent.SERVICE,
      'Checking embedding service health',
      context,
    );

    try {
      const response = await axios.get<EmbeddingServiceHealthResponse>(
        `${this.serviceUrl}/health`,
        {
          timeout: this.timeoutConfig.healthCheckTimeoutMs,
          headers: {
            'X-Request-ID': requestId,
          },
        },
        // Use configured health check timeout
      );

      embeddingsLogger.debug(
        EmbeddingComponent.SERVICE,
        `Health check successful: ${response.data.status}`,
        context,
        response.data,
      );

      embeddingsLogger.removeContext(requestId);
      return response.data;
    } catch (error: any) {
      const categorizedError = this.categorizeError(error);

      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Health check failed: ${categorizedError.message}`,
        context,
        {
          serviceUrl: this.serviceUrl,
          isNetworkError: categorizedError.isNetworkError,
          isServiceError: categorizedError.isServiceError,
          statusCode: categorizedError.statusCode,
          responseData: categorizedError.responseData,
        },
      );

      embeddingsLogger.removeContext(requestId);
      throw categorizedError;
    }
  }

  /**
   * Get embeddings for multiple files
   * @param imagePaths Array of file paths to process
   * @param rawPaths Optional original paths for mapping response
   * @param timeoutMs Optional override for the base timeout
   * @returns ClipCache with embeddings
   */
  public async getEmbeddings(
    imagePaths: string[],
    requestId: string,
    rawPaths?: string[],
    timeoutMs?: number,
  ): Promise<ClipCache> {
    // Update base timeout if provided
    if (timeoutMs) {
      this.baseTimeoutMs = timeoutMs;
      this.timeoutConfig.baseTimeoutMs = timeoutMs;
    }

    const originalPaths = rawPaths?.slice() ?? imagePaths.slice();
    const startTime = Date.now();

    const context = embeddingsLogger.createContext({
      requestId,
      mediaCount: imagePaths.length,
      source: 'getEmbeddings',
    });

    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      `HTTP client processing ${imagePaths.length} files`,
      context,
      { pathCount: imagePaths.length },
    );

    // Process files one by one to avoid overwhelming the server
    // In the future, we could implement concurrent processing with a configurable limit
    const clipCache: ClipCache = {};

    for (let i = 0; i < imagePaths.length; i++) {
      const filePath = imagePaths[i];
      const originalPath = originalPaths[i];

      try {
        embeddingsLogger.debug(
          EmbeddingComponent.SERVICE,
          `Processing file ${i + 1}/${imagePaths.length}: ${filePath}`,
          context,
          { filePath, progress: { current: i + 1, total: imagePaths.length } },
        );

        // Get file metadata
        const metadata = await this.getFileMetadata(filePath);

        // Calculate appropriate timeout based on file metadata
        const calculatedTimeout = this.calculateTimeout({
          fileSize: metadata.fileSize,
          mediaType: metadata.mediaType,
          duration: metadata.duration,
        });

        embeddingsLogger.debug(
          EmbeddingComponent.SERVICE,
          `Calculated timeout for ${filePath}: ${calculatedTimeout}ms`,
          context,
          {
            filePath,
            fileSize: metadata.fileSize,
            mediaType: metadata.mediaType,
            calculatedTimeout,
          },
        );

        // Upload the file and get embedding with dynamic timeout
        const response = await this.uploadFileWithRetry(
          filePath,
          requestId,
          calculatedTimeout,
          metadata,
        );

        // Transform Python's snake_case metadata to camelCase for consistency
        const mappedMetadata = mapPythonMetadata(response.debugMetadata);

        // Construct the cache entry
        const cacheEntry: ClipCacheEntry = {
          schemaVersion: '1.0.0',
          filePath: originalPath, // Use the original path as the key
          mediaType: metadata.mediaType,
          mtime: metadata.mtime,
          fileSize: metadata.fileSize,
          dimensions: metadata.dimensions,
          duration: metadata.duration,
          embedding: response.embedding,
          embeddingModel: response.debugMetadata?.model || 'openai/clip-vit-base-patch32',
          embeddingConfig: {
            // Use transformed camelCase keys
            augmentation:
              typeof mappedMetadata.enableAugmentation === 'boolean'
                ? mappedMetadata.enableAugmentation
                : undefined,
            numFrames:
              typeof mappedMetadata.numFrames === 'number' ? mappedMetadata.numFrames : null,
            samplingMethod:
              typeof mappedMetadata.methodUsed === 'string' ? mappedMetadata.methodUsed : undefined,
            // Include other relevant config fields
            deviceType: mappedMetadata.device,
            batchId: mappedMetadata.batchId,
          },
          processingTimestamp: response.debugMetadata?.timestamp || new Date().toISOString(),
          debugMetadata: {
            // Include both original snake_case and transformed camelCase metadata for compatibility
            ...response.debugMetadata,
            // Add transformed keys
            ...mappedMetadata,
            // Add client-side information
            calculatedTimeout,
            clientProcessingTime: Date.now() - startTime,
            // Include processing statistics
            fileSize: metadata.fileSize,
            mediaType: metadata.mediaType,
          },
          error: response.error ?? undefined,
          detail: response.detail ?? undefined,
        };

        // Validate against the schema
        if (!validateClipCacheEntry(cacheEntry)) {
          const validationErrors = validateClipCacheEntry.errors || [];
          embeddingsLogger.warn(
            EmbeddingComponent.VALIDATION,
            `ClipCacheEntry validation failed for ${filePath}`,
            context,
            {
              validationErrors,
              filePath,
              cacheEntry: JSON.stringify(cacheEntry).substring(0, 200) + '...', // Log truncated entry
            },
          );

          // Attempt to fix common validation issues
          this.fixCommonValidationIssues(cacheEntry);
        }

        clipCache[originalPath] = cacheEntry;

        embeddingsLogger.debug(
          EmbeddingComponent.SERVICE,
          `Successfully processed ${filePath}`,
          context,
          {
            filePath,
            embeddingSize: response.embedding?.length || 0,
            processingTimeMs: mappedMetadata.processingTimeMs,
            timeout: calculatedTimeout,
          },
        );
      } catch (error: any) {
        // Extract timeout information if this was a timeout error
        const isTimeoutError =
          error.code === 'ECONNABORTED' || (error.message && error.message.includes('timeout'));

        const categorizedError = this.categorizeError(error);

        embeddingsLogger.error(
          EmbeddingComponent.SERVICE,
          `Failed to process file ${filePath}: ${categorizedError.message}`,
          context,
          {
            filePath,
            error: categorizedError,
            isNetworkError: categorizedError.isNetworkError,
            isServiceError: categorizedError.isServiceError,
            isTimeoutError,
            statusCode: categorizedError.statusCode,
          },
        );

        // Create appropriate error message based on error type
        let errorMessage = '';
        let errorDetail = '';

        if (isTimeoutError) {
          errorMessage = `Timeout error: The embedding service took too long to respond. The file may be too large or complex to process within the configured timeout.`;
          errorDetail = categorizedError.message;
        } else if (categorizedError.isNetworkError) {
          errorMessage = `Network error: Cannot connect to embedding service - the embedding service might be down or unreachable`;
          errorDetail = categorizedError.message;
        } else if (categorizedError.isServiceError) {
          errorMessage = `Service error: The embedding service responded with an error (${categorizedError.statusCode})`;
          errorDetail = categorizedError.responseData?.detail || categorizedError.message;
        } else {
          errorMessage = `Processing error: ${categorizedError.message}`;
          errorDetail = categorizedError.originalError?.stack || '';
        }

        // Create error entry that conforms to the schema
        const errorEntry: ClipCacheEntry = {
          schemaVersion: '1.0.0',
          filePath: originalPath,
          mediaType: 'image', // Default if metadata extraction fails
          mtime: 0,
          fileSize: 0,
          dimensions: { width: 1, height: 1 },
          duration: null,
          embedding: [], // Empty array for failed embeddings
          embeddingModel: 'unknown',
          embeddingConfig: {},
          processingTimestamp: new Date().toISOString(),
          error: errorMessage,
          detail: errorDetail,
          debugMetadata: {
            errorType: isTimeoutError
              ? 'timeout_error'
              : categorizedError.isNetworkError
                ? 'network_error'
                : categorizedError.isServiceError
                  ? 'service_error'
                  : 'processing_error',
            statusCode: categorizedError.statusCode,
            errorTime: new Date().toISOString(),
          },
        };

        // Validate error entry
        if (!validateClipCacheEntry(errorEntry)) {
          embeddingsLogger.warn(
            EmbeddingComponent.VALIDATION,
            `Error entry validation failed for ${filePath}`,
            context,
            { validationErrors: validateClipCacheEntry.errors, filePath },
          );
          this.fixCommonValidationIssues(errorEntry);
        }

        clipCache[originalPath] = errorEntry;
      }
    }

    const totalTime = Date.now() - startTime;
    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      `Completed processing ${imagePaths.length} files in ${totalTime}ms`,
      context,
      { fileCount: imagePaths.length, totalTimeMs: totalTime },
    );

    embeddingsLogger.removeContext(requestId);
    return clipCache;
  }

  /**
   * Upload a file to the embedding service and get the embedding
   * @param filePath Path to the file
   * @param requestId Request ID for logging
   * @param timeoutMs Timeout in milliseconds
   * @param metadata Optional file metadata for retry timeout adjustments
   * @returns Embedding response
   */
  private async uploadFileWithRetry(
    filePath: string,
    requestId: string,
    timeoutMs: number,
    metadata?: {
      fileSize: number;
      filename: string; // Add filename
      mediaType: 'image' | 'video';
      duration?: number | null;
    },
  ): Promise<EmbeddingResponse> {
    let lastError: Error | null = null;
    let currentTimeout = timeoutMs;
    for (let attempt = 1; attempt <= this.maxRetries + 1; attempt++) {
      try {
        if (attempt > 1) {
          embeddingsLogger.debug(
            EmbeddingComponent.SERVICE,
            `Retry attempt ${attempt}/${this.maxRetries + 1} for ${filePath}`,
            requestId,
            {
              filePath,
              retryCount: attempt,
              timeout: currentTimeout,
            },
          );

          // Exponential backoff with jitter
          const delay = this.retryDelayMs * Math.pow(2, attempt - 1) * (0.5 + Math.random() * 0.5);
          await new Promise((resolve) => setTimeout(resolve, delay));

          // Increase timeout for retries - give more time on each retry attempt
          if (metadata) {
            // For retries, increase timeout by 50% each time
            currentTimeout = Math.min(currentTimeout * 1.5, this.timeoutConfig.maxTimeoutMs);

            embeddingsLogger.debug(
              EmbeddingComponent.SERVICE,
              `Increased timeout for retry: ${currentTimeout}ms`,
              requestId,
              { currentTimeout, retryNumber: attempt },
            );
          }
        }

        // Determine MIME type from metadata.filename or path
        const fileExtension = path.extname(metadata?.filename || filePath).toLowerCase();
        let contentType = 'application/octet-stream';
        if (metadata?.mediaType === 'image') {
          if (['.jpg', '.jpeg'].includes(fileExtension)) contentType = 'image/jpeg';
          else if (fileExtension === '.png') contentType = 'image/png';
          else if (fileExtension === '.webp') contentType = 'image/webp';
        } else if (metadata?.mediaType === 'video') {
          if (fileExtension === '.mp4') contentType = 'video/mp4';
          else if (fileExtension === '.webm') contentType = 'video/webm';
          else if (fileExtension === '.mov') contentType = 'video/quicktime';
        }

        // sanitize filename for HTTP header
        const rawFilename = metadata?.filename || path.basename(filePath);
        const safeFilename = encodeURIComponent(rawFilename);

        const headers: Record<string, string> = {
          'Content-Type': contentType,
          'X-Request-ID': requestId,
          'X-Filename': safeFilename,
          'X-Media-Type':
            metadata?.mediaType ||
            (fileExtension &&
            ['.mp4', '.webm', '.mov', '.avi', '.mkv', '.wmv', '.m4v'].includes(fileExtension)
              ? 'video'
              : 'image'),
          ...(metadata?.mediaType === 'video' &&
            metadata.duration && { 'X-Video-Duration': String(metadata.duration) }),
        };

        // Always use stream for upload
        const fileStream = createReadStream(filePath);
        embeddingsLogger.debug(EmbeddingComponent.SERVICE, 'Uploading file stream...', requestId, {
          filePath,
          size: metadata?.fileSize,
          timeout: currentTimeout,
          contentType,
        });
        const response = await axios.post<EmbeddingResponse>(
          `${this.serviceUrl}/embed`,
          fileStream,
          {
            headers: { ...headers },
            timeout: currentTimeout,
          },
        );

        return response.data;
      } catch (error: any) {
        lastError = error;
        // ...existing retry and error handling...
      }
    }

    // Transform the last error into a categorized error if it's not already one
    if (lastError && !(lastError instanceof EmbeddingServiceError)) {
      lastError = this.categorizeError(lastError);
    }

    throw lastError || new Error(`Failed to upload file after ${this.maxRetries} attempts`);
  }

  /**
   * Extract metadata from a file
   * @param filePath Path to the file
   * @returns File metadata
   */
  private async getFileMetadata(filePath: string): Promise<{
    mtime: number;
    fileSize: number;
    dimensions: { width: number; height: number };
    duration: number | null;
    mediaType: 'image' | 'video';
    filename: string; // Add filename
  }> {
    try {
      // Get file stats
      const stats = await fs.stat(filePath);

      // Determine media type based on extension
      const ext = path.extname(filePath).toLowerCase();
      const mediaType = ['.mp4', '.mov', '.webm', '.avi', '.mkv', '.wmv', '.m4v'].includes(ext)
        ? 'video'
        : 'image';

      // Default values for dimensions
      const dimensions = { width: 1, height: 1 };
      let duration: number | null = null;

      // For videos, use ffprobe to get accurate duration and dimensions
      if (mediaType === 'video') {
        try {
          // Use ffprobe to get video metadata
          const ffprobeResult = await execFileAsync('ffprobe', [
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

          const metadata = JSON.parse(ffprobeResult.stdout);
          if (
            metadata?.streams?.length > 0 &&
            metadata.streams[0].width &&
            metadata.streams[0].height
          ) {
            dimensions.width = parseInt(metadata.streams[0].width, 10);
            dimensions.height = parseInt(metadata.streams[0].height, 10);
          }

          if (metadata?.streams?.length > 0 && metadata.streams[0].duration) {
            duration = parseFloat(metadata.streams[0].duration);
            embeddingsLogger.debug(
              EmbeddingComponent.SERVICE,
              `Extracted video duration: ${duration}s for ${filePath}`,
              undefined,
              { duration, filePath },
            );
          }
        } catch (ffprobeError: any) {
          embeddingsLogger.warn(
            EmbeddingComponent.SERVICE,
            `Failed to extract video metadata with ffprobe: ${ffprobeError.message}`,
            undefined,
            { filePath, error: ffprobeError },
          );
          // Continue with default values if ffprobe fails
        }
      } else if (mediaType === 'image') {
        try {
          // Optimization: Read only necessary bytes for image-size detection
          const buffer = Buffer.alloc(8192); // Increased buffer size for larger image headers
          const fd = await fs.open(filePath, 'r');
          await fd.read(buffer, 0, 8192, 0);
          await fd.close();

          const imageInfo = imageSize(buffer);
          if (imageInfo?.width && imageInfo?.height) {
            dimensions.width = imageInfo.width;
            dimensions.height = imageInfo.height;
          }
        } catch (imgErr: any) {
          embeddingsLogger.warn(
            EmbeddingComponent.SERVICE,
            `Failed to get image dimensions: ${imgErr.message}`,
            undefined,
            { filePath, error: imgErr },
          );
          // Continue with default values if image size extraction fails
        }
      }

      return {
        mtime: stats.mtimeMs,
        fileSize: stats.size,
        dimensions,
        duration,
        mediaType,
        filename: path.basename(filePath), // Add filename
      };
    } catch (error: any) {
      embeddingsLogger.warn(
        EmbeddingComponent.SERVICE,
        `Failed to get metadata for ${filePath}: ${error.message}`,
        undefined,
        { filePath, error },
      );

      // Return default values if metadata extraction fails
      return {
        mtime: 0,
        fileSize: 0,
        dimensions: { width: 1, height: 1 },
        duration: null,
        mediaType: 'image', // Default to image
        filename: path.basename(filePath), // Add filename
      };
    }
  }

  /**
   * Fix common validation issues that might occur in ClipCacheEntry objects
   * @param entry The ClipCacheEntry to fix
   */
  private fixCommonValidationIssues(entry: ClipCacheEntry): void {
    // Ensure embedding is an array
    if (!Array.isArray(entry.embedding)) {
      entry.embedding = [];
    }

    // Ensure dimensions has required properties
    if (!entry.dimensions || typeof entry.dimensions !== 'object') {
      entry.dimensions = { width: 1, height: 1 };
    } else {
      if (typeof entry.dimensions.width !== 'number' || entry.dimensions.width < 1) {
        entry.dimensions.width = 1;
      }
      if (typeof entry.dimensions.height !== 'number' || entry.dimensions.height < 1) {
        entry.dimensions.height = 1;
      }
    }

    // Ensure embeddingConfig is an object
    if (!entry.embeddingConfig || typeof entry.embeddingConfig !== 'object') {
      entry.embeddingConfig = {};
    }

    // Ensure required string fields have values
    if (!entry.schemaVersion || typeof entry.schemaVersion !== 'string') {
      entry.schemaVersion = '1.0.0';
    }

    if (!entry.filePath || typeof entry.filePath !== 'string') {
      entry.filePath = 'unknown_path';
    }

    if (!entry.embeddingModel || typeof entry.embeddingModel !== 'string') {
      entry.embeddingModel = 'unknown';
    }

    // Ensure processingTimestamp is a valid date-time string
    if (!entry.processingTimestamp || typeof entry.processingTimestamp !== 'string') {
      entry.processingTimestamp = new Date().toISOString();
    } else {
      try {
        // Verify it's a valid date
        const date = new Date(entry.processingTimestamp);
        if (isNaN(date.getTime())) {
          entry.processingTimestamp = new Date().toISOString();
        }
      } catch (error: any) {
        embeddingsLogger.warn(
          EmbeddingComponent.SERVICE,
          `Failed to validate processingTimestamp for ${entry.filePath}: ${error.message}`,
          undefined,
          { filePath: entry.filePath, error },
        );
        entry.processingTimestamp = new Date().toISOString();
      }
    }

    // Ensure mediaType is a valid enum value
    if (entry.mediaType !== 'image' && entry.mediaType !== 'video') {
      // default to file extension
      const ext = path.extname(entry.filePath).toLowerCase();
      entry.mediaType = ['.mp4', '.mov', '.webm', '.avi', '.mkv', '.wmv', '.m4v'].includes(ext)
        ? 'video'
        : 'image';
    }
  }
}
