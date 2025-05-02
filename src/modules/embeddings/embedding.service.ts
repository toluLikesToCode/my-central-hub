/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unused-vars */
// modules/embeddings/embedding.service.ts
import { spawn, ChildProcessWithoutNullStreams, execFile, execSync } from 'child_process'; // Import execSync
import Ajv, { ValidateFunction } from 'ajv';
import addFormats from 'ajv-formats';
import clipCacheSchema from '../../../schemas/clipCache.schema.json'; // Adjust path if needed
import path from 'path';
import fs from 'fs/promises'; // Use fs/promises for async file operations
import { promisify } from 'util'; // Needed for promisifying execFile if not using fs/promises directly
import { imageSize } from 'image-size'; // Import image-size correctly
import {
  Logger,
  ConsoleTransport,
  FileTransport,
  JsonFormatter,
  PrettyFormatter,
} from '../../utils/logger';
import { config } from '../../config/server.config'; // Assuming config for paths

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
  duration: number | null; // Made explicitly number | null
  embedding: number[];
  embeddingModel: string;
  embeddingConfig: {
    numFrames?: number | null;
    augmentation?: boolean;
    samplingMethod?: string;
    [k: string]: unknown;
  };
  processingTimestamp: string; // ISO 8601 date-time string
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
  // Ensure the definition path is correct within your schema file
  if (!clipCacheSchema.definitions || !(clipCacheSchema.definitions as any).ClipCacheEntry) {
    throw new Error(
      'Schema definitions or ClipCacheEntry definition missing in clipCache.schema.json',
    );
  }
  validateEntry = ajv.compile<ClipCacheEntry>((clipCacheSchema.definitions as any).ClipCacheEntry);
} catch (err: any) {
  console.error('FATAL: Failed to compile ClipCacheEntry JSON Schema:', err);
  // Depending on desired behavior, you might exit or use a dummy validator
  // Using a dummy validator that always fails ensures no invalid data passes:
  validateEntry = ((data: any) => {
    (validateEntry as any).errors = [{ message: 'Schema compilation failed' }];
    return false;
  }) as ValidateFunction<ClipCacheEntry>;
  // Alternatively, exit if schema validation is critical: process.exit(1);
}

// --- Logging Setup --- //
const NODE_LOG_PREFIX = '[NodeEmbeddingService]';
// NOTE: Consider moving log file path to config/server.config.ts
// Log file path now comes from config
const LOG_FILE_PATH = path.resolve(
  config.logging.logDir, // Use centralized log directory from config
  'embedding_service.log',
);
// Ensure log directory exists
try {
  fs.mkdir(path.dirname(LOG_FILE_PATH), { recursive: true });
} catch (e) {
  console.error('Error creating log directory:', e);
}

const logger = new Logger({
  transports: [
    new ConsoleTransport({
      formatter: new PrettyFormatter({
        useColors: true,
        useBoxes: true,
        showTimestamp: true,
      }),
      level: config.logging.level || 'info', // Use level from config or default
    }),
    new FileTransport({
      filename: LOG_FILE_PATH,
      formatter: new JsonFormatter(),
      level: 'debug', // Keep file log level potentially more verbose
    }),
  ],
});

// Add a dedicated logger for failed validation attempts
const FAILED_VALIDATION_LOG_PATH = path.resolve(config.logging.logDir, 'failed_validation.log');
const failedValidationLogger = new Logger({
  transports: [
    new FileTransport({
      filename: FAILED_VALIDATION_LOG_PATH,
      formatter: new JsonFormatter(),
      level: 'error',
    }),
  ],
  level: 'error',
});

// --- Configuration --- //
// Read Python settings from the centralized config object
const PYTHON_EXECUTABLE = config.embedding.pythonExecutable;
const PYTHON_SCRIPT_PATH = config.embedding.pythonScriptPath;
const PYTHON_MODEL_ARGS: string[] = config.embedding?.modelArgs || [];
const INACTIVITY_TIMEOUT_MS = config.embedding?.inactivityTimeoutMs || 5 * 60 * 1000; // 5 minutes default
const PYTHON_SCRIPT_TIMEOUT_MS = config.embedding?.scriptTimeoutMs || 15 * 60 * 1000; // 15 minutes default

// --- Types (Internal) --- //
interface EmbeddingResponseFromPython {
  // Structure expected directly from the Python script's JSON output per file
  embedding?: number[];
  error?: string;
  detail?: string;
  debugMetadata?: any;
}

interface FileMetadata {
  mtime: number;
  fileSize: number;
  dimensions: { width: number; height: number };
  duration: number | null;
  mediaType: 'image' | 'video';
}

interface EmbeddingRequestInternal {
  paths: string[];
  resolve: (result: ClipCache) => void; // Resolve with ClipCache structure
  reject: (error: Error) => void;
  startTime: number; // Track start time for logging duration
  timeoutHandle?: NodeJS.Timeout; // Store timeout handle
}

// --- Status Types (Exported) --- //
export type EmbeddingServiceState = 'IDLE' | 'PROCESSING' | 'STARTING' | 'ERROR' | 'STOPPED';
export interface EmbeddingServiceStatus {
  state: EmbeddingServiceState;
  pid: number | null;
  isStarting: boolean;
  isProcessing: boolean;
  queueLength: number;
  currentBatch?: { count: number; total: number; current: string };
  lastError?: string; // Made optional
}

// Promisify execFile for ffprobe
const execFileAsync = promisify(execFile);

/**
 * Manages a persistent Python child process for CLIP embedding generation.
 * Handles spawning, communication (stdin/stdout), metadata fetching, validation,
 * and error handling/restarts.
 */
class EmbeddingService {
  private pythonProcess: ChildProcessWithoutNullStreams | null = null;
  private isStarting = false;
  private isStopping = false; // Flag to prevent restarts during manual stop
  private requestQueue: EmbeddingRequestInternal[] = [];
  private currentProcessing: EmbeddingRequestInternal | null = null;
  private responseBuffer = '';
  private inactivityTimer: NodeJS.Timeout | null = null;
  private lastProgress: { processed: number; total: number; current: string } | null = null;
  private lastError: string | null = null;

  constructor() {
    logger.info(`${NODE_LOG_PREFIX} Initializing Embedding Service...`);
    this.validateConfig();
    this.checkDependencies(); // Check for ffprobe on startup
    this.setupExitHandlers();
    // Do not start the Python process immediately; wait for the first request.
  }

  private validateConfig() {
    // Basic checks for essential config/paths
    if (!PYTHON_EXECUTABLE)
      logger.warn(`${NODE_LOG_PREFIX} PYTHON_EXECUTABLE not set, defaulting.`);
    try {
      fs.access(PYTHON_SCRIPT_PATH, fs.constants.R_OK); // Check if script is readable
    } catch (e) {
      logger.error(
        `${NODE_LOG_PREFIX} Python script not found or not readable at: ${PYTHON_SCRIPT_PATH}`,
      );
      this.lastError = `Python script not accessible at ${PYTHON_SCRIPT_PATH}`;
      // Consider preventing service start if script is missing
    }
  }

  private checkDependencies() {
    // Check for ffprobe
    try {
      execSync('ffprobe -version', { stdio: 'ignore' }); // Execute command, ignore output
      logger.info(`${NODE_LOG_PREFIX} Dependency check: ffprobe found.`);
    } catch (error) {
      logger.error(
        `${NODE_LOG_PREFIX} Dependency check failed: ffprobe not found in PATH. Video metadata extraction will fail.`,
      );
      // Consider setting an error state or warning prominently
    }
    // Add checks for ffmpeg if needed by python script logic too?
  }

  // --- Python Process Management --- //

  private resetInactivityTimer() {
    this.clearInactivityTimer();
    // Only set timer if process exists and we are not deliberately stopping it
    if (this.pythonProcess && !this.isStopping) {
      this.inactivityTimer = setTimeout(() => {
        this.stopDueToInactivity();
      }, INACTIVITY_TIMEOUT_MS);
      // Allow Node.js to exit if this timer is the only thing active
      if (this.inactivityTimer.unref) this.inactivityTimer.unref();
    }
  }

  private clearInactivityTimer() {
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }
  }

  private stopDueToInactivity() {
    if (this.isStopping) return; // Already stopping
    logger.info(`${NODE_LOG_PREFIX} Stopping Python process due to inactivity.`);
    this.isStopping = true; // Mark as stopping to prevent auto-restart
    this.stop(); // Use the main stop method
  }

  private async startPythonProcess(): Promise<void> {
    if (this.pythonProcess || this.isStarting) {
      logger.warn(`${NODE_LOG_PREFIX} Process already running or starting.`);
      return Promise.resolve(); // Don't reject, just return
    }
    if (this.lastError === `Python script not accessible at ${PYTHON_SCRIPT_PATH}`) {
      logger.error(`${NODE_LOG_PREFIX} Cannot start process, script is inaccessible.`);
      return Promise.reject(new Error(this.lastError));
    }

    this.isStarting = true;
    this.isStopping = false; // Reset stopping flag
    this.lastError = null; // Clear previous error
    logger.info(
      `${NODE_LOG_PREFIX} Starting Python process: ${PYTHON_EXECUTABLE} "${PYTHON_SCRIPT_PATH}" ${PYTHON_MODEL_ARGS.join(' ')}`,
    );

    return new Promise((resolve, reject) => {
      try {
        // Ensure script path is quoted if it contains spaces
        this.pythonProcess = spawn(PYTHON_EXECUTABLE, [PYTHON_SCRIPT_PATH, ...PYTHON_MODEL_ARGS], {
          stdio: ['pipe', 'pipe', 'pipe'], // stdin, stdout, stderr
        });

        this.isStarting = false; // Process spawned, not necessarily fully ready, but starting phase over

        this.pythonProcess.stdout.on('data', (data: Buffer) => {
          // Optimization: Decode buffer only once
          const chunk = data.toString('utf-8');
          logger.debug(`${NODE_LOG_PREFIX} [PYTHON STDOUT RAW] ${chunk.length} chars`);
          this.responseBuffer += chunk;
          this.processResponseBuffer(); // Process lines efficiently
        });

        this.pythonProcess.stderr.on('data', (data: Buffer) => {
          const lines = data.toString('utf-8').split('\n');
          lines.forEach((line) => {
            const trimmed = line.trim();
            if (!trimmed) return; // Skip empty lines

            if (trimmed.startsWith('PROGRESS:')) {
              try {
                const json = trimmed.substring(9).trim(); // More robust substring
                const progress = JSON.parse(json);
                if (progress && typeof progress === 'object') {
                  this.lastProgress = {
                    processed: Number(progress.processed) || 0,
                    total: Number(progress.total) || 0,
                    current: String(progress.current || ''),
                  };
                } else {
                  logger.warn(`${NODE_LOG_PREFIX} Invalid progress JSON structure: ${json}`);
                }
              } catch (e) {
                logger.warn(
                  `${NODE_LOG_PREFIX} Failed to parse progress line: "${trimmed}", Error: ${(e as Error).message}`,
                );
              }
            } else {
              // Log other stderr lines as errors from Python script
              logger.error(`${NODE_LOG_PREFIX} [PYTHON STDERR] ${trimmed}`);
            }
          });
        });

        this.pythonProcess.on('error', (err) => {
          logger.error(`${NODE_LOG_PREFIX} Python process spawn error: ${err.message}`);
          this.lastError = err.message;
          const startError = new Error(`Python process failed to spawn: ${err.message}`);
          this.handleProcessExit(startError); // Pass error for rejection
          reject(startError); // Reject the start promise
        });

        this.pythonProcess.on('exit', (code, signal) => {
          const exitMsg = `Python process exited (Code: ${code}, Signal: ${signal})`;
          logger.warn(`${NODE_LOG_PREFIX} ${exitMsg}`);
          // Only set lastError if it exited unexpectedly (non-zero code, or signal)
          if (code !== 0 || signal) {
            this.lastError = exitMsg;
          }
          this.handleProcessExit(new Error(exitMsg)); // Pass error for rejection
          // Do not reject the start promise here if it already resolved
        });

        logger.info(`${NODE_LOG_PREFIX} Python process started (PID: ${this.pythonProcess.pid}).`);
        this.resetInactivityTimer(); // Start tracking activity
        this.processQueue(); // Process any queued requests
        resolve(); // Resolve the start promise
      } catch (error: any) {
        logger.error(`${NODE_LOG_PREFIX} Failed to spawn Python process: ${error.message}`);
        this.isStarting = false;
        this.pythonProcess = null;
        const spawnError = new Error(`Failed to spawn Python process: ${error.message}`);
        this.lastError = spawnError.message;
        this.rejectQueue(spawnError); // Reject queued items
        reject(spawnError); // Reject the start promise
      }
    });
  }

  /** Efficiently process the response buffer line by line */
  private processResponseBuffer() {
    let newlineIndex;
    // Use a loop for efficiency if multiple lines arrive in one chunk
    while ((newlineIndex = this.responseBuffer.indexOf('\n')) >= 0) {
      const jsonResponse = this.responseBuffer.substring(0, newlineIndex).trim();
      // Advance the buffer past the processed line and newline character
      this.responseBuffer = this.responseBuffer.substring(newlineIndex + 1);
      if (jsonResponse) {
        logger.debug(
          `${NODE_LOG_PREFIX} [PYTHON RESPONSE] Processing response line (${jsonResponse.length} chars)`,
        );
        this.handlePythonJsonResponse(jsonResponse); // Handle the parsed line
      }
    }
  }

  private handleProcessExit(error?: Error): void {
    const pid = this.pythonProcess?.pid;
    logger.debug(`${NODE_LOG_PREFIX} handleProcessExit called (PID: ${pid})`);
    this.pythonProcess = null; // Mark process as gone
    this.isStarting = false; // Ensure starting flag is reset
    this.clearInactivityTimer();

    // If there was an active request, reject it
    if (this.currentProcessing) {
      const exitError = error || new Error('Python embedding process exited unexpectedly.');
      logger.error(
        `${NODE_LOG_PREFIX} Python process exited while processing request for ${this.currentProcessing.paths.length} paths.`,
      );
      this.currentProcessing.reject(exitError);
      // Clear timeout associated with this request
      if (this.currentProcessing.timeoutHandle) clearTimeout(this.currentProcessing.timeoutHandle);
      this.currentProcessing = null;
    }

    // Reject all remaining queued requests
    const queueError = error || new Error('Python embedding process is not available.');
    this.rejectQueue(queueError);

    // Conditionally restart if not manually stopped
    if (!this.isStopping) {
      logger.info(`${NODE_LOG_PREFIX} Attempting to restart Python process in 5 seconds...`);
      // Use setTimeout directly, no need for async/await here
      setTimeout(() => {
        logger.debug(`${NODE_LOG_PREFIX} Restart timer fired.`);
        this.startPythonProcess().catch((err) => {
          logger.error(`${NODE_LOG_PREFIX} Auto-restart failed: ${err.message}`);
          // Keep lastError updated if restart fails
          this.lastError = `Auto-restart failed: ${err.message}`;
        });
      }, 5000);
    } else {
      logger.info(
        `${NODE_LOG_PREFIX} Manual stop initiated, Python process will not be restarted.`,
      );
      this.isStopping = false; // Reset flag after handling exit during stop
    }
  }

  // --- Metadata Fetching --- //

  /** Fetches metadata for a single file. */
  private async getFileMetadata(filePath: string): Promise<FileMetadata> {
    let mtime = 0; // Default to 0 for consistency if stat fails
    let fileSize = 0;
    let dimensions = { width: 1, height: 1 }; // Default dimension
    let duration: number | null = null;
    let mediaType: 'image' | 'video' = 'image'; // Default assumption

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
          const dim = imageSize(buffer); // Pass buffer
          dimensions = { width: dim?.width ?? 1, height: dim?.height ?? 1 };
        } catch (imgErr) {
          logger.warn(
            `${NODE_LOG_PREFIX} Failed to get image dimensions for ${filePath}: ${(imgErr as Error).message}. Using default 1x1.`,
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
            logger.warn(`${NODE_LOG_PREFIX} ffprobe found no video stream info for ${filePath}.`);
          }
        } catch (ffprobeErr) {
          logger.warn(
            `${NODE_LOG_PREFIX} ffprobe failed for ${filePath}: ${(ffprobeErr as Error).message}.`,
          );
          // Keep default dimensions/duration
        }
      }
    } catch (statErr) {
      logger.warn(
        `${NODE_LOG_PREFIX} Failed to stat file ${filePath}: ${(statErr as Error).message}. Using default metadata.`,
      );
      // Keep default mtime/size if stat fails, might indicate file removed
      // Should we propagate this error more clearly?
    }

    return { mtime, fileSize, dimensions, duration, mediaType };
  }

  /** Fetches metadata for multiple files concurrently. */
  private async getBatchMetadata(filePaths: string[]): Promise<Record<string, FileMetadata>> {
    const metadataPromises = filePaths.map((fp) => this.getFileMetadata(fp));
    const results = await Promise.allSettled(metadataPromises);
    const metadataMap: Record<string, FileMetadata> = {};

    results.forEach((result, index) => {
      const filePath = filePaths[index];
      if (result.status === 'fulfilled') {
        metadataMap[filePath] = result.value;
      } else {
        // Log error but still provide a default entry so processing can continue
        logger.error(
          `${NODE_LOG_PREFIX} Failed to get metadata for ${filePath} in batch: ${result.reason?.message || result.reason}`,
        );
        metadataMap[filePath] = {
          // Provide default/fallback metadata
          mtime: 0,
          fileSize: 0,
          dimensions: { width: 1, height: 1 },
          duration: null,
          mediaType: path.extname(filePath).match(/\.(mp4|mov|webm)$/i) ? 'video' : 'image', // Best guess
        };
      }
    });
    return metadataMap;
  }

  // --- Response Handling & Validation --- //

  private async handlePythonJsonResponse(jsonResponse: string): Promise<void> {
    if (!this.currentProcessing) {
      logger.warn(`${NODE_LOG_PREFIX} Received response from Python but no request is processing.`);
      return;
    }

    const requestStartTime = this.currentProcessing.startTime;
    const currentRequest = this.currentProcessing; // Capture ref in case it changes
    this.currentProcessing = null; // Mark as done processing *before* async metadata fetching

    try {
      const pythonOutput: Record<string, EmbeddingResponseFromPython> = JSON.parse(jsonResponse);
      const filePathsInResponse = Object.keys(pythonOutput);
      logger.debug(
        `${NODE_LOG_PREFIX} Parsed Python response for ${filePathsInResponse.length} files.`,
      );

      // --- Optimization: Fetch metadata concurrently for all files in the batch ---
      logger.debug(
        `${NODE_LOG_PREFIX} Fetching metadata for ${filePathsInResponse.length} files...`,
      );
      const batchMetadata = await this.getBatchMetadata(filePathsInResponse);
      logger.debug(`${NODE_LOG_PREFIX} Finished fetching metadata.`);

      const finalResults: ClipCache = {}; // Build the response object conforming to ClipCache

      for (const filePath of filePathsInResponse) {
        const pyEntry = pythonOutput[filePath];
        const meta = batchMetadata[filePath]; // Get pre-fetched metadata

        if (!meta) {
          logger.error(
            `${NODE_LOG_PREFIX} Metadata missing for ${filePath} after batch fetch. Skipping.`,
          );
          // Create an error entry?
          finalResults[filePath] = {
            schemaVersion: '1.0.0',
            filePath: filePath,
            error: 'Metadata fetch failed',
            // Add other required fields with defaults if possible, or make them optional in schema
            mtime: 0,
            fileSize: 0,
            dimensions: { width: 1, height: 1 },
            duration: null,
            mediaType: 'image',
            embedding: [],
            embeddingModel: 'unknown',
            embeddingConfig: {},
            processingTimestamp: new Date().toISOString(),
          } as ClipCacheEntry; // May fail validation if embedding is required
          continue;
        }

        // Extract model/config from debugMetadata if present
        const debug = pyEntry.debugMetadata || {};
        const embeddingModel = String(debug.model || config.embedding?.defaultModel || 'unknown');
        const embeddingConfig: ClipCacheEntry['embeddingConfig'] = {
          augmentation:
            typeof debug.enable_augmentation === 'boolean' ? debug.enable_augmentation : undefined,
          numFrames: typeof debug.num_frames === 'number' ? debug.num_frames : null,
          samplingMethod: typeof debug.method_used === 'string' ? debug.method_used : undefined,
        };

        // Construct the entry attempting to match the schema
        const entryData: Partial<ClipCacheEntry> = {
          schemaVersion: '1.0.0',
          filePath: filePath,
          embedding: pyEntry.embedding, // Will be validated later
          debugMetadata: pyEntry.debugMetadata,
          error: pyEntry.error != null ? String(pyEntry.error) : undefined,
          detail: pyEntry.detail != null ? String(pyEntry.detail) : undefined,
          processingTimestamp: new Date().toISOString(),
          mtime: meta.mtime,
          fileSize: meta.fileSize,
          dimensions: meta.dimensions,
          mediaType: meta.mediaType,
          duration: meta.duration,
          embeddingModel,
          embeddingConfig,
        };

        // Handle case where embedding failed in Python
        if (entryData.error && !entryData.embedding) {
          // Schema requires embedding. Set to empty array to pass validation,
          // client should check for error field.
          entryData.embedding = [];
        } else if (!entryData.embedding && !entryData.error) {
          // No embedding and no error? Treat as error.
          entryData.error = 'Embedding missing without error from Python';
          entryData.embedding = [];
        }

        // ---> VALIDATE the constructed entry against the schema <---
        if (validateEntry(entryData)) {
          // If valid, assign the validated (and now typed) entry
          finalResults[filePath] = entryData as ClipCacheEntry;
        } else {
          // If invalid, log details and store an error-focused object
          const validationErrors = JSON.stringify(validateEntry.errors);
          logger.error(
            `${NODE_LOG_PREFIX} Constructed cache entry failed validation for: ${filePath}`,
            {
              constructedData: entryData, // Log data before validation
              pythonData: pyEntry, // Log raw python data
              errors: validationErrors,
            },
          );
          // Also log to failed_validation.log
          failedValidationLogger.error(`Failed validation for: ${filePath}`, {
            constructedData: entryData,
            pythonData: pyEntry,
            errors: validationErrors,
          });

          // Create a minimal structure indicating validation failure
          // This structure *must* still pass basic validation if possible,
          // or the client needs specific handling for these error objects.
          finalResults[filePath] = {
            schemaVersion: '1.0.0',
            filePath: filePath,
            error: 'Internal schema validation failed',
            detail: validationErrors,
            // Add required fields with placeholder/default values
            mtime: meta.mtime, // Use fetched meta even on validation error
            fileSize: meta.fileSize,
            dimensions: meta.dimensions,
            duration: meta.duration,
            mediaType: meta.mediaType,
            embedding: [], // Empty embedding on validation error
            embeddingModel: embeddingModel,
            embeddingConfig: embeddingConfig,
            processingTimestamp: entryData.processingTimestamp || new Date().toISOString(),
          } as ClipCacheEntry; // Note: This might still fail if required fields missing
        }
      }

      // Resolve the original promise with the processed results
      const duration = Date.now() - requestStartTime;
      logger.info(
        `${NODE_LOG_PREFIX} Successfully processed batch of ${filePathsInResponse.length} paths in ${duration} ms.`,
      );
      currentRequest.resolve(finalResults);
      if (currentRequest.timeoutHandle) clearTimeout(currentRequest.timeoutHandle); // Clear timeout on success
    } catch (e: any) {
      logger.error(
        `${NODE_LOG_PREFIX} Failed to parse/process JSON response from Python: ${e.message}. Response: ${jsonResponse}`,
        e,
      );
      const processingError = new Error(`Failed to process response from Python: ${e.message}`);
      currentRequest.reject(processingError); // Reject the original promise
      if (currentRequest.timeoutHandle) clearTimeout(currentRequest.timeoutHandle); // Clear timeout on error
      this.lastError = processingError.message; // Update last error
    } finally {
      // Ensure we attempt to process the queue regardless of success/failure of this batch
      this.processQueue();
      this.resetInactivityTimer(); // Reset timer after processing a response
    }
  }

  // --- Request Queuing and Processing --- //

  private rejectQueue(error: Error): void {
    if (this.requestQueue.length > 0) {
      logger.warn(
        `${NODE_LOG_PREFIX} Rejecting ${this.requestQueue.length} queued request(s) due to error: ${error.message}`,
      );
      this.requestQueue.forEach((req) => {
        if (req.timeoutHandle) clearTimeout(req.timeoutHandle); // Clear individual timeouts
        req.reject(error);
      });
      this.requestQueue = []; // Clear the queue
    }
  }

  private processQueue(): void {
    if (
      this.currentProcessing ||
      this.requestQueue.length === 0 ||
      !this.pythonProcess ||
      this.isStarting
    ) {
      logger.debug(
        `${NODE_LOG_PREFIX} Skipping processQueue (Processing: ${!!this.currentProcessing}, Queue: ${this.requestQueue.length}, Proc: ${!!this.pythonProcess}, Starting: ${this.isStarting})`,
      );
      return; // Process busy, queue empty, or process not ready/starting
    }

    this.currentProcessing = this.requestQueue.shift()!; // Get next request from queue
    this.lastProgress = null; // Reset progress for new batch
    logger.info(
      `${NODE_LOG_PREFIX} Sending batch of ${this.currentProcessing.paths.length} paths to Python (Queue: ${this.requestQueue.length}).`,
    );

    const requestPayload = { imagePaths: this.currentProcessing.paths }; // Python script expects this structure

    try {
      // Add newline delimiter for Python script's readline()
      const requestJson = JSON.stringify(requestPayload) + '\n';
      this.responseBuffer = ''; // Clear buffer before sending new request

      // Handle potential write errors (e.g., process died between check and write)
      if (!this.pythonProcess?.stdin?.writable) {
        throw new Error('Python process stdin is not writable.');
      }

      this.pythonProcess.stdin.write(requestJson, (err) => {
        if (err) {
          logger.error(`${NODE_LOG_PREFIX} Failed to write to Python stdin: ${err.message}`);
          // Process might be dead, trigger exit handling
          const writeError = new Error(`Failed to send data to Python: ${err.message}`);
          this.currentProcessing?.reject(writeError);
          if (this.currentProcessing?.timeoutHandle)
            clearTimeout(this.currentProcessing.timeoutHandle);
          this.currentProcessing = null;
          // Don't necessarily kill here, let exit handler manage potential restart
          this.handleProcessExit(writeError);
        } else {
          logger.debug(`${NODE_LOG_PREFIX} Data written to Python stdin successfully.`);
          this.resetInactivityTimer(); // Reset timer after successful write
        }
      });
    } catch (error: any) {
      logger.error(`${NODE_LOG_PREFIX} Error writing to Python stdin: ${error.message}`);
      const catchError = new Error(`Error sending data to Python: ${error.message}`);
      this.currentProcessing.reject(catchError);
      if (this.currentProcessing.timeoutHandle) clearTimeout(this.currentProcessing.timeoutHandle);
      this.currentProcessing = null;
      // Trigger exit handling if write fails critically
      this.handleProcessExit(catchError);
    }
  }

  /**
   * Public method to request embeddings. Starts Python process if needed.
   */
  public async getEmbeddings(
    imagePaths: string[],
    timeoutMs = PYTHON_SCRIPT_TIMEOUT_MS,
  ): Promise<ClipCache> {
    // Return ClipCache structure

    // Start process if it's not running and not already stopping/starting
    if (!this.pythonProcess && !this.isStarting && !this.isStopping) {
      logger.info(`${NODE_LOG_PREFIX} Python process not running. Starting for new request...`);
      try {
        await this.startPythonProcess();
      } catch (startErr: any) {
        logger.error(
          `${NODE_LOG_PREFIX} Failed to start Python process for request: ${startErr.message}`,
        );
        // Reject immediately if start failed
        return Promise.reject(new Error(`Failed to start Python process: ${startErr.message}`));
      }
    } else {
      // If process exists, reset inactivity timer as a request is coming in
      this.resetInactivityTimer();
    }

    this.lastError = null; // Clear last error on new request attempt

    return new Promise((resolve, reject) => {
      let settled = false;
      const startTime = Date.now();

      const request: EmbeddingRequestInternal = {
        paths: imagePaths,
        startTime: startTime,
        resolve: (result: ClipCache) => {
          // Expect ClipCache
          if (settled) return;
          settled = true;
          if (request.timeoutHandle) clearTimeout(request.timeoutHandle);
          this.resetInactivityTimer(); // Reset timer on successful completion
          resolve(result);
        },
        reject: (error: Error) => {
          if (settled) return;
          settled = true;
          if (request.timeoutHandle) clearTimeout(request.timeoutHandle);
          logger.error(
            `${NODE_LOG_PREFIX} Request failed after ${Date.now() - startTime} ms: ${error.message}`,
          );
          reject(error);
        },
      };

      // Setup timeout for *this specific request*
      request.timeoutHandle = setTimeout(() => {
        if (settled) return;
        logger.warn(
          `${NODE_LOG_PREFIX} Request timed out after ${timeoutMs} ms for ${imagePaths.length} paths.`,
        );
        // Remove request from queue *if it's still there*
        const index = this.requestQueue.findIndex((r) => r === request);
        if (index > -1) {
          this.requestQueue.splice(index, 1);
          logger.debug(`${NODE_LOG_PREFIX} Removed timed-out request from queue.`);
        } else if (this.currentProcessing === request) {
          // If it was actively processing, we can't easily abort Python,
          // but we should reject the promise and nullify currentProcessing
          // so the queue can potentially continue. Maybe kill python? Risky.
          logger.error(
            `${NODE_LOG_PREFIX} Request timed out while actively processing. Python process might be stuck.`,
          );
          this.currentProcessing = null; // Allow queue to proceed, previous request is lost
          this.lastError = `Request timed out while processing ${imagePaths.length} paths.`;
          // Consider killing and restarting python process here if it's stuck
          this.stop(); // Force stop and restart cycle
        }
        request.reject(new Error(`Embedding request timed out after ${timeoutMs} ms.`));
      }, timeoutMs);

      // Add to queue and attempt processing
      this.requestQueue.push(request);
      logger.debug(
        `${NODE_LOG_PREFIX} Queued request for ${imagePaths.length} paths. Queue size: ${this.requestQueue.length}`,
      );
      // Trigger queue processing immediately if possible
      if (!this.currentProcessing && this.pythonProcess && !this.isStarting) {
        this.processQueue();
      }
    });
  }

  // --- Service Control --- //

  /** Manually stops the Python process and rejects pending requests. */
  public stop(): void {
    logger.info(`${NODE_LOG_PREFIX} Manual stop requested.`);
    this.isStopping = true; // Prevent restarts during manual stop
    this.clearInactivityTimer(); // Stop inactivity timer

    if (this.pythonProcess) {
      logger.info(`${NODE_LOG_PREFIX} Killing Python process (PID: ${this.pythonProcess.pid})...`);
      this.pythonProcess.kill(); // Send SIGTERM
      this.pythonProcess = null; // Assume it will exit
    } else {
      logger.info(`${NODE_LOG_PREFIX} Python process already stopped.`);
    }

    // Reject current and queued requests
    const stopError = new Error('Embedding service is stopping.');
    if (this.currentProcessing) {
      this.currentProcessing.reject(stopError);
      if (this.currentProcessing.timeoutHandle) clearTimeout(this.currentProcessing.timeoutHandle);
      this.currentProcessing = null;
    }
    this.rejectQueue(stopError);
    // isStopping will be reset by handleProcessExit if it triggers,
    // or reset on next successful start
  }

  public getStatus(): EmbeddingServiceStatus {
    let state: EmbeddingServiceState = 'IDLE';
    if (this.isStopping)
      state = 'STOPPED'; // Explicitly stopped state
    else if (this.isStarting) state = 'STARTING';
    else if (!this.pythonProcess && this.lastError)
      state = 'ERROR'; // Error state if process down + error exists
    else if (!this.pythonProcess && !this.lastError)
      state = 'STOPPED'; // Stopped cleanly or hasn't started
    else if (this.currentProcessing) state = 'PROCESSING';
    // IDLE = process running but no current task

    const status: EmbeddingServiceStatus = {
      state,
      pid: this.pythonProcess?.pid ?? null,
      isStarting: this.isStarting,
      isProcessing: !!this.currentProcessing,
      queueLength: this.requestQueue.length,
      currentBatch: this.lastProgress
        ? {
            count: this.lastProgress.processed,
            total: this.lastProgress.total,
            current: this.lastProgress.current,
          }
        : undefined,
      lastError: this.lastError || undefined,
    };
    return status;
  }

  private setupExitHandlers() {
    // Graceful shutdown: Ensure Python process is killed when Node exits
    const handleExit = () => {
      logger.info(`${NODE_LOG_PREFIX} Node process exiting. Stopping Python process...`);
      this.isStopping = true; // Prevent restarts during shutdown
      this.stop();
    };
    process.on('exit', handleExit);
    // Handle Ctrl+C, kill, etc.
    process.on('SIGINT', () => {
      logger.info(`${NODE_LOG_PREFIX} Received SIGINT.`);
      handleExit();
      process.exit(0); // Exit Node process after cleanup attempt
    });
    process.on('SIGTERM', () => {
      logger.info(`${NODE_LOG_PREFIX} Received SIGTERM.`);
      handleExit();
      process.exit(0); // Exit Node process after cleanup attempt
    });
    process.on('uncaughtException', (err) => {
      logger.child([err.stack]).error(`${NODE_LOG_PREFIX} Uncaught Exception: ${err.message}`);
      // Optionally try to stop python before exiting
      handleExit();
      process.exit(1); // Exit with error code
    });
    process.on('unhandledRejection', (reason, promise) => {
      logger.error(`${NODE_LOG_PREFIX} Unhandled Rejection at: ${promise}, reason: ${reason}`);
      // Optionally try to stop python before exiting
      handleExit();
      process.exit(1); // Exit with error code
    });
  }
}

// Export a singleton instance
export const embeddingService = new EmbeddingService();
