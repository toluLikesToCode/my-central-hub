/**
 * src/modules/embeddings/embedding.service.ts
 * This file handles the embedding service logic for processing embeddings.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unused-vars */
import { spawn, ChildProcessWithoutNullStreams, execFile, execSync } from 'child_process';
import Ajv, { ValidateFunction } from 'ajv';
import addFormats from 'ajv-formats';
import clipCacheSchema from '../../../schemas/clipCache.schema.json';
import path from 'path';
import fs from 'fs/promises';
import { promisify } from 'util';
import { imageSize } from 'image-size';
import { embeddingsLogger, EmbeddingComponent, EmbeddingContext } from './embeddingsLogger';
import { config } from '../../config/server.config';
import { randomUUID } from 'crypto';

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
const PYTHON_EXECUTABLE = config.embedding.pythonExecutable;
const PYTHON_SCRIPT_PATH = config.embedding.pythonScriptPath;
const PYTHON_MODEL_ARGS: string[] = config.embedding?.modelArgs || [];
const INACTIVITY_TIMEOUT_MS = config.embedding?.inactivityTimeoutMs || 5 * 60 * 1000; // 5 minutes default
const PYTHON_SCRIPT_TIMEOUT_MS = config.embedding?.scriptTimeoutMs || 15 * 60 * 1000; // 15 minutes default

// --- Types (Internal) --- //
interface EmbeddingResponseFromPython {
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
  resolve: (result: ClipCache) => void;
  reject: (error: Error) => void;
  startTime: number;
  timeoutHandle?: NodeJS.Timeout;
  requestId: string;
  batchId?: string;
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
 * Manages a persistent Python child process for CLIP embedding generation.
 * Handles spawning, communication (stdin/stdout), metadata fetching, validation,
 * and error handling/restarts.
 */
class EmbeddingService {
  private pythonProcess: ChildProcessWithoutNullStreams | null = null;
  private isStarting = false;
  private isStopping = false;
  private requestQueue: EmbeddingRequestInternal[] = [];
  private currentProcessing: EmbeddingRequestInternal | null = null;
  private responseBuffer = '';
  private inactivityTimer: NodeJS.Timeout | null = null;
  private lastProgress: { processed: number; total: number; current: string } | null = null;
  private lastError: string | null = null;

  // --- Restart Backoff State ---
  private restartAttempts = 0;
  private nextRestartDelayMs = 5000; // Initial delay
  private readonly MAX_RESTART_ATTEMPTS = 5;
  private readonly BASE_RESTART_DELAY_MS = 5000;
  private readonly MAX_RESTART_DELAY_MS = 60 * 1000; // Maximum delay (1 minute)

  constructor() {
    embeddingsLogger.info(EmbeddingComponent.SERVICE, 'Initializing Embedding Service...');

    this.validateConfig();
    this.checkDependencies();
    this.setupExitHandlers();
    // Do not start the Python process immediately; wait for the first request.
  }

  private validateConfig() {
    // Basic checks for essential config/paths
    if (!PYTHON_EXECUTABLE)
      embeddingsLogger.warn(EmbeddingComponent.SERVICE, 'PYTHON_EXECUTABLE not set, defaulting.');

    try {
      fs.access(PYTHON_SCRIPT_PATH, fs.constants.R_OK);
    } catch (e) {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Python script not found or not readable at: ${PYTHON_SCRIPT_PATH}`,
        undefined,
        { scriptPath: PYTHON_SCRIPT_PATH },
      );
      this.lastError = `Python script not accessible at ${PYTHON_SCRIPT_PATH}`;
    }
  }

  private checkDependencies() {
    // Check for ffprobe
    try {
      execSync('ffprobe -version', { stdio: 'ignore' });
      embeddingsLogger.info(EmbeddingComponent.SERVICE, 'Dependency check: ffprobe found.');
    } catch (error) {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        'Dependency check failed: ffprobe not found in PATH. Video metadata extraction will fail.',
        undefined,
        { error },
      );
    }
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
    embeddingsLogger.info(EmbeddingComponent.SERVICE, 'Stopping Python process due to inactivity.');
    this.isStopping = true;
    this.stop();
  }

  public async startPythonProcess(): Promise<void> {
    if (this.pythonProcess || this.isStarting) {
      embeddingsLogger.warn(EmbeddingComponent.SERVICE, 'Process already running or starting.');
      return Promise.resolve();
    }

    if (this.lastError === `Python script not accessible at ${PYTHON_SCRIPT_PATH}`) {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        'Cannot start process, script is inaccessible.',
        undefined,
        { scriptPath: PYTHON_SCRIPT_PATH },
      );
      return Promise.reject(new Error(this.lastError));
    }

    this.isStarting = true;
    this.isStopping = false;
    this.lastError = null;

    embeddingsLogger.info(
      EmbeddingComponent.SERVICE,
      `Starting Python process: ${PYTHON_EXECUTABLE} "${PYTHON_SCRIPT_PATH}" ${PYTHON_MODEL_ARGS.join(' ')}`,
      undefined,
      {
        pythonExecutable: PYTHON_EXECUTABLE,
        scriptPath: PYTHON_SCRIPT_PATH,
        modelArgs: PYTHON_MODEL_ARGS,
      },
    );

    return new Promise((resolve, reject) => {
      try {
        this.pythonProcess = spawn(PYTHON_EXECUTABLE, [PYTHON_SCRIPT_PATH, ...PYTHON_MODEL_ARGS], {
          stdio: ['pipe', 'pipe', 'pipe'], // stdin, stdout, stderr
        });

        // Defer clearing isStarting until the process actually spawns
        this.pythonProcess.once('spawn', () => {
          this.isStarting = false;
          embeddingsLogger.info(
            EmbeddingComponent.SERVICE,
            `Python process spawn event received (PID: ${this.pythonProcess?.pid}).`,
            undefined,
            { pid: this.pythonProcess?.pid },
          );

          // Reset restart counters on successful spawn
          this.restartAttempts = 0;
          this.nextRestartDelayMs = this.BASE_RESTART_DELAY_MS;
          embeddingsLogger.debug(EmbeddingComponent.SERVICE, 'Restart counters reset.');
        });

        this.pythonProcess.stdout.on('data', (data: Buffer) => {
          // Optimization: Decode buffer only once
          const chunk = data.toString('utf-8');
          embeddingsLogger.debug(
            EmbeddingComponent.SERVICE,
            `PYTHON STDOUT RAW received (${chunk.length} chars)`,
            this.currentProcessing?.requestId,
            { byteLength: data.length },
          );

          this.responseBuffer += chunk;
          this.processResponseBuffer();
        });

        this.pythonProcess.stderr.on('data', (data: Buffer) => {
          const lines = data.toString('utf-8').split('\n');
          lines.forEach((line) => {
            const trimmed = line.trim();
            if (!trimmed) return; // Skip empty lines

            if (trimmed.startsWith('PROGRESS:')) {
              try {
                const json = trimmed.substring(9).trim();
                const progress = JSON.parse(json);
                if (progress && typeof progress === 'object') {
                  this.lastProgress = {
                    processed: Number(progress.processed) || 0,
                    total: Number(progress.total) || 0,
                    current: String(progress.current || ''),
                  };

                  embeddingsLogger.pythonErr(trimmed, this.currentProcessing?.requestId, {
                    progress: this.lastProgress,
                    batchId: this.currentProcessing?.batchId,
                  });
                } else {
                  embeddingsLogger.warn(
                    EmbeddingComponent.PYTHON,
                    `Invalid progress JSON structure: ${json}`,
                    this.currentProcessing?.requestId,
                  );
                }
              } catch (e) {
                embeddingsLogger.warn(
                  EmbeddingComponent.PYTHON,
                  `Failed to parse progress line: "${trimmed}"`,
                  this.currentProcessing?.requestId,
                  { error: e },
                );
              }
            } else {
              // Log other stderr lines through the Python error handler
              embeddingsLogger.pythonErr(trimmed, this.currentProcessing?.requestId);
            }
          });
        });

        this.pythonProcess.on('error', (err) => {
          embeddingsLogger.error(
            EmbeddingComponent.SERVICE,
            `Python process spawn error: ${err.message}`,
            this.currentProcessing?.requestId,
            { error: err },
          );

          this.lastError = err.message;
          const startError = new Error(`Python process failed to spawn: ${err.message}`);
          this.handleProcessExit(startError);
          reject(startError);
        });

        this.pythonProcess.on('exit', (code, signal) => {
          const exitMsg = `Python process exited (Code: ${code}, Signal: ${signal})`;
          embeddingsLogger.warn(
            EmbeddingComponent.SERVICE,
            exitMsg,
            this.currentProcessing?.requestId,
            { exitCode: code, exitSignal: signal },
          );

          // Only set lastError if it exited unexpectedly (non-zero code, or signal)
          if (code !== 0 || signal) {
            this.lastError = exitMsg;
          }
          this.handleProcessExit(new Error(exitMsg));
        });

        embeddingsLogger.info(
          EmbeddingComponent.SERVICE,
          `Python process started (PID: ${this.pythonProcess.pid}).`,
          undefined,
          { pid: this.pythonProcess.pid },
        );

        this.resetInactivityTimer();
        this.processQueue();
        resolve();
      } catch (error: any) {
        embeddingsLogger.error(
          EmbeddingComponent.SERVICE,
          `Failed to spawn Python process: ${error.message}`,
          undefined,
          { error },
        );

        this.isStarting = false;
        this.pythonProcess = null;
        const spawnError = new Error(`Failed to spawn Python process: ${error.message}`);
        this.lastError = spawnError.message;
        this.rejectQueue(spawnError);
        reject(spawnError);
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
        // Only attempt JSON parse on lines that look like JSON
        if (jsonResponse.startsWith('{') || jsonResponse.startsWith('[')) {
          embeddingsLogger.debug(
            EmbeddingComponent.SERVICE,
            `Processing response line (${jsonResponse.length} chars)`,
            this.currentProcessing?.requestId,
          );
          this.handlePythonJsonResponse(jsonResponse);
        } else {
          embeddingsLogger.debug(
            EmbeddingComponent.SERVICE,
            `Skipping non-JSON stdout line: ${jsonResponse}`,
            this.currentProcessing?.requestId,
          );
        }
      }
    }
  }

  private handleProcessExit(error?: Error): void {
    const pid = this.pythonProcess?.pid;
    embeddingsLogger.debug(
      EmbeddingComponent.SERVICE,
      `handleProcessExit called (PID: ${pid})`,
      this.currentProcessing?.requestId,
      { pid },
    );

    this.pythonProcess = null;
    this.isStarting = false;
    this.clearInactivityTimer();

    // If there was an active request, reject it
    if (this.currentProcessing) {
      const exitError = error || new Error('Python embedding process exited unexpectedly.');
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Python process exited while processing request for ${this.currentProcessing.paths.length} paths.`,
        this.currentProcessing.requestId,
        {
          pathCount: this.currentProcessing.paths.length,
          error: exitError,
        },
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
      if (this.restartAttempts >= this.MAX_RESTART_ATTEMPTS) {
        const maxAttemptMsg = `Maximum restart attempts (${this.MAX_RESTART_ATTEMPTS}) reached. Service will remain in ERROR state.`;
        embeddingsLogger.error(EmbeddingComponent.SERVICE, maxAttemptMsg, undefined, {
          attempts: this.restartAttempts,
          maxAttempts: this.MAX_RESTART_ATTEMPTS,
        });

        this.lastError = this.lastError ? `${this.lastError}. ${maxAttemptMsg}` : maxAttemptMsg;
      } else {
        this.restartAttempts++;
        const currentDelay = this.nextRestartDelayMs;

        embeddingsLogger.info(
          EmbeddingComponent.SERVICE,
          `Attempting restart ${this.restartAttempts}/${this.MAX_RESTART_ATTEMPTS} in ${currentDelay / 1000} seconds...`,
          undefined,
          {
            attempt: this.restartAttempts,
            maxAttempts: this.MAX_RESTART_ATTEMPTS,
            delayMs: currentDelay,
          },
        );

        setTimeout(() => {
          embeddingsLogger.debug(
            EmbeddingComponent.SERVICE,
            `Restart timer fired for attempt ${this.restartAttempts}.`,
          );

          this.startPythonProcess().catch((err) => {
            embeddingsLogger.error(
              EmbeddingComponent.SERVICE,
              `Auto-restart attempt ${this.restartAttempts} failed: ${err.message}`,
              undefined,
              {
                attempt: this.restartAttempts,
                error: err,
              },
            );

            this.lastError = `Auto-restart attempt ${this.restartAttempts} failed: ${err.message}`;
          });
        }, currentDelay);

        // Exponential backoff for next attempt
        this.nextRestartDelayMs = Math.min(this.nextRestartDelayMs * 2, this.MAX_RESTART_DELAY_MS);
      }
    } else {
      embeddingsLogger.info(
        EmbeddingComponent.SERVICE,
        'Manual stop initiated, Python process will not be restarted.',
      );

      this.isStopping = false; // Reset flag after handling exit during stop
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
        embeddingsLogger.error(
          EmbeddingComponent.SERVICE,
          `Failed to get metadata for ${filePath} in batch`,
          this.currentProcessing?.requestId,
          {
            filePath,
            error: result.reason,
          },
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
      embeddingsLogger.warn(
        EmbeddingComponent.SERVICE,
        'Received response from Python but no request is processing.',
      );
      return;
    }

    const requestStartTime = this.currentProcessing.startTime;
    const currentRequest = this.currentProcessing;
    const requestId = currentRequest.requestId;
    this.currentProcessing = null; // Mark as done processing *before* async metadata fetching

    try {
      const pythonOutput: Record<string, EmbeddingResponseFromPython> = JSON.parse(jsonResponse);
      const filePathsInResponse = Object.keys(pythonOutput);

      embeddingsLogger.debug(
        EmbeddingComponent.SERVICE,
        `Parsed Python response for ${filePathsInResponse.length} files.`,
        requestId,
        { fileCount: filePathsInResponse.length },
      );

      // --- Optimization: Fetch metadata concurrently for all files in the batch ---
      const timerEnd = embeddingsLogger.startTimer('fetchMetadata', requestId);
      embeddingsLogger.debug(
        EmbeddingComponent.SERVICE,
        `Fetching metadata for ${filePathsInResponse.length} files...`,
        requestId,
      );

      const batchMetadata = await this.getBatchMetadata(filePathsInResponse);

      timerEnd(); // Log the duration
      embeddingsLogger.debug(EmbeddingComponent.SERVICE, `Finished fetching metadata.`, requestId);

      const finalResults: ClipCache = {};

      for (const filePath of filePathsInResponse) {
        const pyEntry = pythonOutput[filePath];
        const meta = batchMetadata[filePath]; // Get pre-fetched metadata

        if (!meta) {
          embeddingsLogger.error(
            EmbeddingComponent.SERVICE,
            `Metadata missing for ${filePath} after batch fetch. Skipping.`,
            requestId,
            { filePath },
          );

          // Create an error entry
          finalResults[filePath] = {
            schemaVersion: '1.0.0',
            filePath: filePath,
            error: 'Metadata fetch failed',
            mtime: 0,
            fileSize: 0,
            dimensions: { width: 1, height: 1 },
            duration: null,
            mediaType: 'image',
            embedding: [],
            embeddingModel: 'unknown',
            embeddingConfig: {},
            processingTimestamp: new Date().toISOString(),
          } as ClipCacheEntry;
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
          embedding: pyEntry.embedding,
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

          embeddingsLogger.logValidationIssue(
            `Constructed cache entry failed validation for: ${filePath}`,
            requestId,
            {
              constructedData: entryData,
              pythonData: pyEntry,
              errors: validateEntry.errors,
            },
          );

          // Create a minimal structure indicating validation failure
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
          } as ClipCacheEntry;
        }
      }

      // Resolve the original promise with the processed results
      const duration = Date.now() - requestStartTime;
      embeddingsLogger.info(
        EmbeddingComponent.SERVICE,
        `Successfully processed batch of ${filePathsInResponse.length} paths in ${duration} ms.`,
        requestId,
        {
          fileCount: filePathsInResponse.length,
          durationMs: duration,
          batchId: currentRequest.batchId,
        },
      );

      currentRequest.resolve(finalResults);
      if (currentRequest.timeoutHandle) clearTimeout(currentRequest.timeoutHandle);
    } catch (e: any) {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Failed to parse/process JSON response from Python: ${e.message}`,
        requestId,
        {
          error: e,
          responsePreview:
            jsonResponse.length > 200 ? jsonResponse.substring(0, 200) + '...' : jsonResponse,
        },
      );

      const processingError = new Error(`Failed to process response from Python: ${e.message}`);
      currentRequest.reject(processingError);
      if (currentRequest.timeoutHandle) clearTimeout(currentRequest.timeoutHandle);
      this.lastError = processingError.message;
    } finally {
      // Ensure we attempt to process the queue regardless of success/failure of this batch
      this.processQueue();
      this.resetInactivityTimer();
    }
  }

  // --- Request Queuing and Processing --- //

  private rejectQueue(error: Error): void {
    if (this.requestQueue.length > 0) {
      embeddingsLogger.warn(
        EmbeddingComponent.SERVICE,
        `Rejecting ${this.requestQueue.length} queued request(s) due to error`,
        undefined,
        {
          queueLength: this.requestQueue.length,
          error,
        },
      );

      this.requestQueue.forEach((req) => {
        if (req.timeoutHandle) clearTimeout(req.timeoutHandle);
        req.reject(error);
      });

      this.requestQueue = [];
    }
  }

  private processQueue(): void {
    if (
      this.currentProcessing ||
      this.requestQueue.length === 0 ||
      !this.pythonProcess ||
      this.isStarting
    ) {
      embeddingsLogger.debug(
        EmbeddingComponent.SERVICE,
        `Skipping processQueue (Processing: ${!!this.currentProcessing}, Queue: ${this.requestQueue.length}, Proc: ${!!this.pythonProcess}, Starting: ${this.isStarting})`,
      );
      return;
    }

    this.currentProcessing = this.requestQueue.shift()!;
    this.lastProgress = null; // Reset progress for new batch

    const requestId = this.currentProcessing.requestId;
    const batchId = this.currentProcessing.batchId || randomUUID().substring(0, 8);
    this.currentProcessing.batchId = batchId;

    embeddingsLogger.logBatch(batchId, this.currentProcessing.paths.length, requestId, {
      queueRemaining: this.requestQueue.length,
    });

    const requestPayload = { imagePaths: this.currentProcessing.paths };

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
          embeddingsLogger.error(
            EmbeddingComponent.SERVICE,
            `Failed to write to Python stdin: ${err.message}`,
            requestId,
            { error: err },
          );

          // Process might be dead, trigger exit handling
          const writeError = new Error(`Failed to send data to Python: ${err.message}`);
          this.currentProcessing?.reject(writeError);
          if (this.currentProcessing?.timeoutHandle)
            clearTimeout(this.currentProcessing.timeoutHandle);
          this.currentProcessing = null;
          // Don't necessarily kill here, let exit handler manage potential restart
          this.handleProcessExit(writeError);
        } else {
          embeddingsLogger.debug(
            EmbeddingComponent.SERVICE,
            'Data written to Python stdin successfully.',
            requestId,
          );

          this.resetInactivityTimer();
        }
      });
    } catch (error: any) {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Error writing to Python stdin: ${error.message}`,
        requestId,
        { error },
      );

      const catchError = new Error(`Error sending data to Python: ${error.message}`);
      this.currentProcessing.reject(catchError);
      if (this.currentProcessing.timeoutHandle) clearTimeout(this.currentProcessing.timeoutHandle);
      this.currentProcessing = null;
      // Trigger exit handling if write fails critically
      this.handleProcessExit(catchError);
    }
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
   * Public method to request embeddings. Starts Python process if needed.
   * Recursively searches the public directory for each filename in imagePaths.
   * The response will use the original request paths as keys and in filePath fields.
   */
  public async getEmbeddings(
    imagePaths: string[],
    rawPaths?: string[],
    timeoutMs = PYTHON_SCRIPT_TIMEOUT_MS,
  ): Promise<ClipCache> {
    const requestId = randomUUID();
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

    // Start process if it's not running and not already stopping/starting
    if (!this.pythonProcess && !this.isStarting && !this.isStopping) {
      embeddingsLogger.info(
        EmbeddingComponent.SERVICE,
        'Python process not running. Starting for new request...',
        context,
      );

      try {
        await this.startPythonProcess();
      } catch (startErr: any) {
        embeddingsLogger.error(
          EmbeddingComponent.SERVICE,
          `Failed to start Python process for request: ${startErr.message}`,
          context,
          { error: startErr },
        );

        embeddingsLogger.removeContext(requestId);
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
        // Use found absolute paths for processing
        paths: foundPaths,
        startTime: startTime,
        requestId,
        resolve: (result: ClipCache) => {
          // Remap result keys and filePath fields to original request paths
          if (settled) return;
          settled = true;
          if (request.timeoutHandle) clearTimeout(request.timeoutHandle);
          this.resetInactivityTimer();

          // Remap result: for each original request path, use the result for the found path
          const remapped: ClipCache = {};
          for (const reqPath of originalPaths) {
            const found = pathMap[reqPath];
            if (found && result[found]) {
              remapped[reqPath] = { ...result[found], filePath: reqPath };
            } else {
              embeddingsLogger.warn(
                EmbeddingComponent.SERVICE,
                `File not found in results: ${reqPath}`,
                context,
                { filePath: reqPath },
              );

              remapped[reqPath] = {
                schemaVersion: '1.0.0',
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

          embeddingsLogger.info(
            EmbeddingComponent.SERVICE,
            `Request completed successfully in ${Date.now() - startTime}ms`,
            context,
            { durationMs: Date.now() - startTime },
          );

          embeddingsLogger.removeContext(requestId);
          resolve(remapped);
        },
        reject: (error: Error) => {
          if (settled) return;
          settled = true;
          if (request.timeoutHandle) clearTimeout(request.timeoutHandle);

          embeddingsLogger.error(
            EmbeddingComponent.SERVICE,
            `Request failed after ${Date.now() - startTime}ms: ${error.message}`,
            context,
            {
              error,
              durationMs: Date.now() - startTime,
            },
          );

          embeddingsLogger.removeContext(requestId);
          reject(error);
        },
      };

      // Setup timeout for *this specific request*
      request.timeoutHandle = setTimeout(() => {
        if (settled) return;

        embeddingsLogger.warn(
          EmbeddingComponent.SERVICE,
          `Request timed out after ${timeoutMs}ms for ${imagePaths.length} paths.`,
          context,
          {
            timeoutMs,
            pathCount: imagePaths.length,
          },
        );

        // Remove request from queue *if it's still there*
        const index = this.requestQueue.findIndex((r) => r === request);
        if (index > -1) {
          this.requestQueue.splice(index, 1);
          embeddingsLogger.debug(
            EmbeddingComponent.SERVICE,
            'Removed timed-out request from queue.',
            context,
          );
        } else if (this.currentProcessing === request) {
          // If it was actively processing, we can't easily abort Python,
          // but we should reject the promise and nullify currentProcessing
          // so the queue can potentially continue.
          embeddingsLogger.error(
            EmbeddingComponent.SERVICE,
            'Request timed out while actively processing. Python process might be stuck.',
            context,
          );

          this.currentProcessing = null; // Allow queue to proceed
          this.lastError = `Request timed out while processing ${imagePaths.length} paths.`;
          this.processQueue(); // Attempt to process next queued request
        }

        request.reject(new Error(`Embedding request timed out after ${timeoutMs}ms.`));
      }, timeoutMs);

      // Add to queue and attempt processing
      this.requestQueue.push(request);

      embeddingsLogger.debug(
        EmbeddingComponent.SERVICE,
        `Queued request for ${imagePaths.length} paths. Queue size: ${this.requestQueue.length}`,
        context,
        {
          pathCount: imagePaths.length,
          queueSize: this.requestQueue.length,
        },
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
    embeddingsLogger.info(EmbeddingComponent.SERVICE, 'Manual stop requested.');

    this.isStopping = true; // Prevent restarts during manual stop
    this.clearInactivityTimer(); // Stop inactivity timer

    if (this.pythonProcess) {
      embeddingsLogger.info(
        EmbeddingComponent.SERVICE,
        `Killing Python process (PID: ${this.pythonProcess.pid})...`,
        undefined,
        { pid: this.pythonProcess.pid },
      );

      this.pythonProcess.kill(); // Send SIGTERM
      this.pythonProcess = null; // Assume it will exit
    } else {
      embeddingsLogger.info(EmbeddingComponent.SERVICE, 'Python process already stopped.');
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
    // Determine service state based on flags and queue
    let state: EmbeddingServiceState;
    if (this.isStopping) {
      state = 'STOPPED';
    } else if (this.isStarting) {
      state = 'STARTING';
    } else if (!this.pythonProcess) {
      state = this.lastError ? 'ERROR' : 'STOPPED';
    } else if (this.currentProcessing || this.requestQueue.length > 0) {
      state = 'PROCESSING';
    } else {
      state = 'IDLE';
    }

    const status = {
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

  private setupExitHandlers() {
    // Graceful shutdown: Ensure Python process is killed when Node exits
    const handleExit = () => {
      embeddingsLogger.info(
        EmbeddingComponent.SERVICE,
        'Node process exiting. Stopping Python process...',
      );

      this.isStopping = true; // Prevent restarts during shutdown
      this.stop();
    };
    process.on('exit', handleExit);
    // Handle Ctrl+C, kill, etc.
    process.on('SIGINT', () => {
      embeddingsLogger.info(EmbeddingComponent.SERVICE, 'Received SIGINT.');

      handleExit();
      process.exit(0); // Exit Node process after cleanup attempt
    });
    process.on('SIGTERM', () => {
      embeddingsLogger.info(EmbeddingComponent.SERVICE, 'Received SIGTERM.');

      handleExit();
      process.exit(0); // Exit Node process after cleanup attempt
    });
    process.on('uncaughtException', (err) => {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Uncaught Exception: ${err.message}`,
        undefined,
        { error: err, stack: err.stack },
      );

      // Optionally try to stop python before exiting
      handleExit();
      process.exit(1); // Exit with error code
    });
    process.on('unhandledRejection', (reason, promise) => {
      embeddingsLogger.error(
        EmbeddingComponent.SERVICE,
        `Unhandled Rejection at: ${promise}, reason: ${reason}`,
        undefined,
        { reason, promise },
      );

      // Optionally try to stop python before exiting
      handleExit();
      process.exit(1); // Exit with error code
    });
  }
}

// Export a singleton instance
export const embeddingService = new EmbeddingService();
