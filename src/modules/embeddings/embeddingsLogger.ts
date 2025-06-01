/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * src/modules/embeddings/embeddingsLogger.ts
 * A dedicated logger for the embeddings pipeline with consistent formatting and correlation IDs.
 */
import path from 'path';
import { randomUUID } from 'crypto';
import * as fs from 'fs';
import {
  Logger,
  ConsoleTransport,
  FileTransport,
  JsonFormatter,
  PrettyFormatter,
} from '../../utils/logger';

// Define component types for better categorization
export enum EmbeddingComponent {
  SERVICE = 'SERVICE',
  HANDLER = 'HANDLER',
  PYTHON = 'PYTHON',
  VALIDATION = 'VALIDATION',
}

// Define context interface for correlation and tracing
export interface EmbeddingContext {
  requestId: string;
  mediaCount?: number;
  source?: string;
  startTime?: number;
  [key: string]: any;
}

class EmbeddingsLogger {
  private logger: Logger;
  private contexts: Map<string, EmbeddingContext> = new Map();
  private progressTimers: Map<string, NodeJS.Timeout> = new Map();

  constructor() {
    // Create a dedicated logger for embeddings with multiple transports
    this.logger = new Logger({
      transports: [
        // Console for immediate feedback with pretty formatting
        new ConsoleTransport({
          formatter: new PrettyFormatter({
            useBoxes: false,
          }),
          level: 'info',
        }),
        // JSON file logs for structured data and debugging
        new FileTransport({
          filename: 'logs/embeddings.log',
          formatter: new JsonFormatter(),
          level: 'debug',
        }),
        // Separate file for validation issues to make them easier to find
        new FileTransport({
          filename: 'logs/embedding_validation.log',
          formatter: new JsonFormatter(),
          level: 'debug',
        }),
        // Dedicated file for error log endpoint
        new FileTransport({
          filename: 'logs/embedding_error_logs.log',
          formatter: new JsonFormatter(),
          level: 'info',
        }),
      ],
      level: 'debug',
      exitOnError: false,
    });
  }

  /**
   * Create a new context for tracking a request through the system
   */
  createContext(data: Partial<EmbeddingContext> = {}): string {
    const requestId = data.requestId || randomUUID().toString();
    const context: EmbeddingContext = {
      requestId,
      startTime: Date.now(),
      ...data,
    };
    this.contexts.set(requestId, context);
    return requestId;
  }

  /**
   * Update an existing context with new information
   */
  updateContext(requestId: string, data: Partial<EmbeddingContext>): void {
    const existing = this.contexts.get(requestId);
    if (existing) {
      this.contexts.set(requestId, { ...existing, ...data });
    }
  }

  /**
   * Remove a context when a request is complete
   */
  removeContext(requestId: string): void {
    this.contexts.delete(requestId);
    // Clean up any progress timers
    if (this.progressTimers.has(requestId)) {
      clearTimeout(this.progressTimers.get(requestId)!);
      this.progressTimers.delete(requestId);
    }
  }

  /**
   * Get context for a request
   */
  getContext(requestId?: string): EmbeddingContext | undefined {
    if (!requestId) return undefined;
    return this.contexts.get(requestId);
  }

  /**
   * Format metadata with context for logging
   */
  private formatMeta(requestId?: string, meta?: Record<string, any>): Record<string, any> {
    const context = requestId ? this.contexts.get(requestId) : undefined;
    return {
      ...meta,
      ...context,
      requestId,
      timestamp: new Date().toISOString(),
      ...(context?.startTime && { duration_ms: Date.now() - context.startTime }),
    };
  }

  /**
   * Log an info message
   */
  info(
    component: EmbeddingComponent,
    message: string,
    requestId?: string,
    meta?: Record<string, any>,
  ): void {
    this.logger.info(`[${component}] ${message}`, this.formatMeta(requestId, meta));
  }

  /**
   * Log a debug message
   */
  debug(
    component: EmbeddingComponent,
    message: string,
    requestId?: string,
    meta?: Record<string, any>,
  ): void {
    this.logger.debug(`[${component}] ${message}`, this.formatMeta(requestId, meta));
  }

  /**
   * Log an error message
   */
  error(
    component: EmbeddingComponent,
    message: string,
    requestId?: string,
    meta?: Record<string, any>,
  ): void {
    this.logger.error(`[${component}] ${message}`, this.formatMeta(requestId, meta));
  }

  /**
   * Log a warning message
   */
  warn(
    component: EmbeddingComponent,
    message: string,
    requestId?: string,
    meta?: Record<string, any>,
  ): void {
    this.logger.warn(`[${component}] ${message}`, this.formatMeta(requestId, meta));
  }

  /**
   * Specialized log for Python stderr output handling
   * Handles both actual errors and progress messages
   */
  pythonErr(message: string, requestId?: string, meta?: Record<string, any>): void {
    if (message.trim().length === 0) return; // Skip empty lines

    // Filter out some very verbose ffmpeg logs that aren't useful
    if (
      message.includes('non-existing PPS 0 referenced') ||
      message.includes('mmco: unref short failure')
    ) {
      return;
    }

    // Handle progress messages
    if (message.startsWith('PROGRESS:')) {
      try {
        // Progress is already logged specifically through the Python class
        return;
      } catch (e) {
        this.warn(
          EmbeddingComponent.PYTHON,
          `Failed to parse progress message: ${message}`,
          requestId,
          { error: e },
        );
      }
    }

    // Determine log level based on message content
    if (
      message.toLowerCase().includes('error') ||
      message.toLowerCase().includes('exception') ||
      message.toLowerCase().includes('failed')
    ) {
      this.error(EmbeddingComponent.PYTHON, message, requestId, meta);
    } else if (
      message.toLowerCase().includes('warning') ||
      message.toLowerCase().includes('deprecated')
    ) {
      this.warn(EmbeddingComponent.PYTHON, message, requestId, meta);
    } else if (message.toLowerCase().includes('debug')) {
      this.debug(EmbeddingComponent.PYTHON, message, requestId, meta);
    } else {
      // Default to info for general Python output
      this.debug(EmbeddingComponent.PYTHON, message, requestId, meta);
    }
  }

  /**
   * Specialized log for validation issues
   */
  logValidationIssue(message: string, requestId?: string, meta?: Record<string, any>): void {
    this.error(EmbeddingComponent.VALIDATION, message, requestId, meta);

    // Also log to specialized validation log file
    try {
      const formattedMeta = this.formatMeta(requestId, meta);
      const timestamp = new Date().toISOString();
      const logEntry = {
        timestamp,
        level: 'ERROR',
        component: EmbeddingComponent.VALIDATION,
        message,
        ...formattedMeta,
      };
      // Write to separate validation log file for easier tracking
      fs.appendFileSync(
        path.resolve(process.cwd(), 'logs/failed_validation.log'),
        JSON.stringify(logEntry) + '\n',
      );
    } catch (e) {
      // If we can't write to the validation log, just continue
      this.error(EmbeddingComponent.VALIDATION, 'Failed to write to validation log', requestId, {
        error: e,
      });
    }
  }

  /**
   * Log a batch processing start with a unique batch ID
   */
  logBatch(
    batchId: string,
    fileCount: number,
    requestId?: string,
    meta?: Record<string, any>,
  ): void {
    this.info(
      EmbeddingComponent.SERVICE,
      `Processing batch ${batchId} with ${fileCount} files`,
      requestId,
      { ...meta, batchId, fileCount },
    );

    // Set up a timer to log progress if the batch is taking a long time
    if (requestId) {
      this.progressTimers.set(
        requestId,
        setTimeout(() => {
          this.warn(
            EmbeddingComponent.SERVICE,
            `Batch ${batchId} has been processing for over 30 seconds`,
            requestId,
            { batchId, fileCount, timeElapsed: '30s' },
          );
        }, 30000),
      );
    }
  }

  /**
   * Helper to measure and log operation duration
   * Returns a function that, when called, will log the time elapsed
   */
  startTimer(operation: string, requestId?: string): () => number {
    const start = Date.now();
    return () => {
      const duration = Date.now() - start;
      this.debug(
        EmbeddingComponent.SERVICE,
        `Operation '${operation}' completed in ${duration}ms`,
        requestId,
        { operation, duration_ms: duration },
      );
      return duration;
    };
  }
}

// Export singleton instance
export const embeddingsLogger = new EmbeddingsLogger();
