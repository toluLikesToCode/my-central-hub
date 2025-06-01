/**
 * src/modules/embeddings/embeddings.handler.ts
 * This file handles the HTTP requests for the embeddings service.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { Socket } from 'net';
import { IncomingRequest } from '../../entities/http';
import { sendWithContext } from '../../entities/sendResponse'; // Ensure this path is correct
import { ClipCacheEntry, embeddingService } from './embedding.service';
import { embeddingsLogger, EmbeddingComponent } from './embeddingsLogger';
import path from 'path';
// config is not directly used in this refactored handler for path validation,
// but embeddingService will use it.
// import { config } from '../../config/server.config';
import { randomUUID } from 'crypto';
import { getHeader } from '../../utils/httpHelpers';

// Helper to summarize large arrays for logging
function summarizeArray(arr: any[]): string {
  if (!Array.isArray(arr)) return String(arr);
  const len = arr.length;
  if (len === 0) return '[]';
  const preview = arr
    .slice(0, 5)
    .map((x) => (typeof x === 'number' ? Number(x).toFixed(4) : String(x)));
  let min = null,
    max = null;
  try {
    const numericArr = arr.filter((x) => typeof x === 'number');
    if (numericArr.length > 0) {
      min = Math.min(...numericArr);
      max = Math.max(...numericArr);
    }
  } catch {
    // empty catch to handle non-numeric arrays or errors in min/max
  }
  return `[Array(len=${len}${min !== null ? `, min=${min}` : ''}${max !== null ? `, max=${max}` : ''}, preview=[${preview.join(', ')}]...)]`;
}

function summarizeObject(obj: any): any {
  if (Array.isArray(obj)) return summarizeArray(obj);
  if (obj && typeof obj === 'object') {
    const copy: any = {}; // Initialize as an empty object
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        // Ensure it's an own property
        if (Array.isArray(obj[key]) && obj[key].length > 20) {
          copy[key] = summarizeArray(obj[key]);
        } else if (typeof obj[key] === 'object' && obj[key] !== null) {
          copy[key] = summarizeObject(obj[key]); // Recursively summarize
        } else {
          copy[key] = obj[key];
        }
      }
    }
    return copy;
  }
  return obj;
}

export const embeddingsController = {
  /**
   * Handles POST requests to /api/embeddings
   * Expects a JSON body: { "files": ["path1", "path2", ...] } or legacy { "imagePaths": [...] }
   * Returns JSON: { "path1": { "embedding": [...] }, "path2": { "embedding": [...] }, ... }
   */
  async handleEmbeddingsRequest(req: IncomingRequest, sock: Socket) {
    // Disable socket timeout for embeddings requests
    if (typeof sock.setTimeout === 'function') {
      sock.setTimeout(0); // disables timeout for this socket
    }

    const requestId = req.ctx?.requestId?.toString() || randomUUID().toString();
    const context = embeddingsLogger.createContext({
      requestId: requestId,
      source: 'http-request',
    });

    const clientInfo = {
      remoteAddress: sock.remoteAddress,
      remotePort: sock.remotePort,
      localAddress: sock.localAddress,
      localPort: sock.localPort,
    };

    embeddingsLogger.info(EmbeddingComponent.HANDLER, 'Received embedding request', context, {
      bodyLength: req.body?.length || 0,
      clientInfo,
    });

    if (req.method !== 'POST') {
      embeddingsLogger.warn(
        EmbeddingComponent.HANDLER,
        `Method not allowed: ${req.method}`,
        context,
        { method: req.method, clientInfo },
      );
      embeddingsLogger.removeContext(requestId);
      return sendWithContext(
        req,
        sock,
        405,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Method Not Allowed, use POST' }),
      );
    }

    if (!req.body || req.body.length === 0) {
      embeddingsLogger.warn(EmbeddingComponent.HANDLER, 'Request body is empty', context, {
        clientInfo,
      });
      embeddingsLogger.removeContext(requestId);
      return sendWithContext(
        req,
        sock,
        400,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Request body is empty' }),
      );
    }

    try {
      const serviceStatus = await embeddingService.getStatus(requestId);
      if (serviceStatus.state === 'ERROR' || serviceStatus.state === 'STOPPED') {
        embeddingsLogger.error(
          EmbeddingComponent.HANDLER,
          `Cannot process embedding request: Embedding service is ${serviceStatus.state}`,
          context,
          {
            clientInfo,
            serviceStatus,
            lastError: serviceStatus.lastError,
          },
        );
        embeddingsLogger.removeContext(requestId);
        return sendWithContext(
          req,
          sock,
          503,
          { 'Content-Type': 'application/json' },
          JSON.stringify({
            error: 'Embedding service unavailable',
            detail: serviceStatus.lastError || `Service is in ${serviceStatus.state} state`,
          }),
        );
      }

      const body = JSON.parse(req.body.toString('utf-8'));
      let requestedPathsStrings = body.imagePaths ?? body.files;
      // Extract numFrames from body or header, with fallback to undefined
      const numFrames: number | undefined = (() => {
        const raw = body.numFrames ?? getHeader(req, 'X-Num-Frames');
        if (raw == null) return undefined;
        const parsed = typeof raw === 'number' ? raw : parseInt(raw, 10);
        return Number.isInteger(parsed) ? parsed : undefined;
      })();

      embeddingsLogger.debug(EmbeddingComponent.HANDLER, 'Parsed request body', context, {
        body: summarizeObject(body),
        pathsRequestedCount: Array.isArray(requestedPathsStrings)
          ? requestedPathsStrings.length
          : typeof requestedPathsStrings === 'string'
            ? 1
            : 0,
        clientInfo,
      });

      if (typeof requestedPathsStrings === 'string') {
        requestedPathsStrings = [requestedPathsStrings];
      } else if (
        !Array.isArray(requestedPathsStrings) ||
        !requestedPathsStrings.every((p) => typeof p === 'string')
      ) {
        embeddingsLogger.warn(
          EmbeddingComponent.HANDLER,
          'Invalid request format—missing "files" or "imagePaths", or they are not string/array of strings.',
          context,
          { bodySummary: summarizeObject(body), clientInfo },
        );
        embeddingsLogger.removeContext(requestId);
        return sendWithContext(
          req,
          sock,
          400,
          { 'Content-Type': 'application/json' },
          JSON.stringify({
            error:
              'Invalid request format. Expected "files" or "imagePaths" as a string or an array of strings.',
          }),
        );
      }

      if (requestedPathsStrings.length === 0) {
        embeddingsLogger.warn(
          EmbeddingComponent.HANDLER,
          'No file paths provided in "files" or "imagePaths".',
          context,
          { clientInfo },
        );
        embeddingsLogger.removeContext(requestId);
        return sendWithContext(
          req,
          sock,
          400,
          { 'Content-Type': 'application/json' },
          JSON.stringify({ error: 'No file paths provided.' }),
        );
      }

      // Normalize paths for consistency and use these as keys and search terms.
      // embeddingService.findFile will handle the actual resolution against configured media directories.
      const normalizedClientPaths = requestedPathsStrings
        .map((p: string) => p.trim())
        .filter((p: string) => p.length > 0) // Filter out empty strings after trim
        .map((p: string) => path.normalize(p)); // Basic normalization

      if (normalizedClientPaths.length === 0) {
        embeddingsLogger.warn(
          EmbeddingComponent.HANDLER,
          'No valid file paths remaining after trimming and normalization.',
          context,
          { originalPaths: requestedPathsStrings, clientInfo },
        );
        embeddingsLogger.removeContext(requestId);
        return sendWithContext(
          req,
          sock,
          400,
          { 'Content-Type': 'application/json' },
          JSON.stringify({ error: 'No valid file paths provided after normalization.' }),
        );
      }

      embeddingsLogger.info(
        EmbeddingComponent.HANDLER,
        `Handler: Requesting embeddings for ${normalizedClientPaths.length} client-provided (normalized) paths.`,
        context,
        {
          normalizedPathCount: normalizedClientPaths.length,
          clientInfo,
        },
      );

      embeddingsLogger.updateContext(requestId, { mediaCount: normalizedClientPaths.length });

      try {
        // `normalizedClientPaths` are used for searching files (by embeddingService.findFile)
        // AND as the keys in the returned ClipCache.
        const embeddingsResult = await embeddingService.getEmbeddings(
          {
            requestedImagePaths: normalizedClientPaths,
            requestId: requestId,
            rawPaths: normalizedClientPaths,
            numFrames: numFrames,
          }, // Pass numFrames (can be number or undefined)
        );

        embeddingsLogger.info(
          EmbeddingComponent.HANDLER,
          `Sending response for ${normalizedClientPaths.length} paths.`,
          context,
          {
            pathCount: normalizedClientPaths.length,
            hasErrors: Object.values(embeddingsResult).some((entry) => !!entry.error),
            clientInfo,
          },
        );
        embeddingsLogger.removeContext(requestId);
        sendWithContext(
          req,
          sock,
          200,
          { 'Content-Type': 'application/json' },
          JSON.stringify(embeddingsResult),
        );
      } catch (error: any) {
        embeddingsLogger.error(
          EmbeddingComponent.HANDLER,
          `Error getting embeddings from service: ${error.message}`,
          context,
          {
            errorName: error.name,
            errorMessage: error.message,
            errorStack: error.stack,
            clientInfo,
          },
        );

        const errorResponse: Record<string, ClipCacheEntry> = {};
        // Use normalizedClientPaths for keys in the error response for consistency
        for (const keyPath of normalizedClientPaths) {
          errorResponse[keyPath] = {
            schemaVersion: '1.1.0', // Consistent with other services
            filePath: keyPath, // The key itself
            mediaType: 'image', // Default; could be refined if type known before error
            mtime: 0,
            fileSize: 0,
            dimensions: { width: 1, height: 1 },
            duration: null,
            embedding: [],
            embeddingModel: 'unknown',
            embeddingConfig: {},
            processingTimestamp: new Date().toISOString(),
            error: `Service error during embedding generation: ${error.message}`,
            detail: error.stack || 'No additional stack trace available.',
          };
        }
        embeddingsLogger.removeContext(requestId);
        sendWithContext(
          req,
          sock,
          500,
          { 'Content-Type': 'application/json' },
          JSON.stringify(errorResponse),
        );
      }
    } catch (error: any) {
      embeddingsLogger.error(
        EmbeddingComponent.HANDLER,
        `Critical error processing embedding request: ${error.message}`,
        context,
        { errorName: error.name, errorMessage: error.message, errorStack: error.stack, clientInfo },
      );

      const statusCode = error instanceof SyntaxError ? 400 : 500;
      const errorMessage =
        error instanceof SyntaxError
          ? 'Invalid JSON in request body.'
          : 'Failed to process embeddings request.';
      embeddingsLogger.removeContext(requestId);
      sendWithContext(
        req,
        sock,
        statusCode,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: errorMessage, detail: error.message }),
      );
    }
  },

  /**
   * Handles POST requests to /api/embeddings/shutdown
   * Instructs the embedding service to stop gracefully.
   */
  async handleShutdownRequest(req: IncomingRequest, sock: Socket): Promise<void> {
    const requestId = req.ctx?.requestId?.toString() || randomUUID().toString();
    const context = embeddingsLogger.createContext({
      requestId,
      source: 'shutdown-request',
    });

    embeddingsLogger.info(EmbeddingComponent.HANDLER, 'Received shutdown request', context);

    embeddingService.stop(); // Python service manages its own lifecycle; this is for Node.js part.

    embeddingsLogger.info(
      EmbeddingComponent.HANDLER,
      'Embedding service stop initiated (Node.js).',
      context,
    );
    embeddingsLogger.removeContext(requestId);
    sendWithContext(
      req,
      sock,
      200,
      { 'Content-Type': 'application/json' },
      JSON.stringify({ message: 'Embedding service shutdown initiated (Node.js component).' }),
    );
  },

  /**
   * Handles GET requests to /api/embeddings/status
   * Returns JSON: { state, pythonServiceHealth, lastError, isProcessingBatch }
   */
  async handleStatusRequest(req: IncomingRequest, sock: Socket): Promise<void> {
    const requestId = req.ctx?.requestId?.toString() || randomUUID().toString();
    const context = embeddingsLogger.createContext({
      requestId,
      source: 'status-request',
    });

    embeddingsLogger.debug(EmbeddingComponent.HANDLER, 'Received status request', context);

    if (req.method !== 'GET') {
      embeddingsLogger.warn(
        EmbeddingComponent.HANDLER,
        `Method not allowed for status: ${req.method}`,
        context,
        { method: req.method },
      );
      embeddingsLogger.removeContext(requestId);
      return sendWithContext(
        req,
        sock,
        405,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Method Not Allowed' }),
      );
    }

    try {
      const status = await embeddingService.getStatus(requestId);
      // The new embeddingService.getStatus() already handles Python health check.
      // No need to manually "start" Python service here as it's managed independently.

      if (status.state === 'ERROR' && status.lastError) {
        embeddingsLogger.warn(
          EmbeddingComponent.HANDLER,
          `Embedding service is reporting an error state: ${status.lastError}`,
          context,
          { serviceStatus: status },
        );
      }

      embeddingsLogger.debug(
        EmbeddingComponent.HANDLER,
        `Returning status: ${status.state}`,
        context,
        status,
      );
      embeddingsLogger.removeContext(requestId);
      sendWithContext(
        req,
        sock,
        200,
        { 'Content-Type': 'application/json' },
        JSON.stringify(status),
      );
    } catch (error: any) {
      embeddingsLogger.error(
        EmbeddingComponent.HANDLER,
        `Error fetching service status: ${error.message}`,
        context,
        { errorName: error.name, errorMessage: error.message, errorStack: error.stack },
      );
      embeddingsLogger.removeContext(requestId);
      sendWithContext(
        req,
        sock,
        500,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Failed to fetch service status', detail: error.message }),
      );
    }
  },

  /**
   * Handles POST requests to /api/embeddings/error-logs
   * Validates, truncates, logs error logs to a new file, and returns a JSON response.
   */
  async handleErrorLogsRequest(req: IncomingRequest, sock: Socket): Promise<void> {
    const requestId = req.ctx?.requestId?.toString() || randomUUID().toString();
    const context = embeddingsLogger.createContext({
      requestId,
      source: 'error-logs',
    });

    if (req.method !== 'POST') {
      embeddingsLogger.warn(
        EmbeddingComponent.HANDLER,
        `Method not allowed: ${req.method} for error-logs endpoint`,
        context,
      );
      embeddingsLogger.removeContext(requestId);
      return sendWithContext(
        req,
        sock,
        405,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Method Not Allowed, use POST' }),
      );
    }

    if (!req.body || req.body.length === 0) {
      embeddingsLogger.warn(EmbeddingComponent.HANDLER, 'Error log request body is empty', context);
      embeddingsLogger.removeContext(requestId);
      return sendWithContext(
        req,
        sock,
        400,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Request body is empty' }),
      );
    }

    let logs: any[] = [];
    try {
      logs = JSON.parse(req.body.toString('utf-8'));
      if (!Array.isArray(logs)) throw new Error('Request body must be a JSON array');
    } catch (err: any) {
      embeddingsLogger.error(
        EmbeddingComponent.HANDLER,
        'Failed to parse error log request body',
        context,
        { error: err.message },
      );
      embeddingsLogger.removeContext(requestId);
      return sendWithContext(
        req,
        sock,
        400,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Invalid JSON or not an array', detail: err.message }),
      );
    }

    // Truncate long string fields in each log object
    const TRUNCATE_LIMIT = 1024;
    function truncateStrings(obj: any): any {
      if (typeof obj === 'string' && obj.length > TRUNCATE_LIMIT) {
        return obj.slice(0, TRUNCATE_LIMIT) + '... (truncated)';
      } else if (Array.isArray(obj)) {
        return obj.map(truncateStrings);
      } else if (obj && typeof obj === 'object') {
        const out: any = {};
        for (const k in obj) {
          if (Object.prototype.hasOwnProperty.call(obj, k)) {
            out[k] = truncateStrings(obj[k]);
          }
        }
        return out;
      }
      return obj;
    }
    const processedLogs = logs.map(truncateStrings);

    // Write to a dedicated error log file (append as JSONL)
    // Only client logs should be written here: do not log server-side errors or internal events
    const fs = await import('fs/promises');
    const path = await import('path');
    // Use structured JSON instead of line-delimited JSONL
    const logFilePath = path.resolve(process.cwd(), 'logs/embedding_error_logs.json');
    try {
      await fs.mkdir(path.dirname(logFilePath), { recursive: true });
      // Overwrite the file with only the new client logs (do not append)
      // If you want to keep only the latest batch, use writeFile instead of appendFile
      // Build a JSON array with metadata and write as pretty JSON
      const logsToWrite = processedLogs.map((entry) => ({
        ...entry,
        receivedAt: new Date().toISOString(),
        requestId,
      }));
      await fs.writeFile(logFilePath, JSON.stringify(logsToWrite, null, 2), 'utf-8');
    } catch (err: any) {
      embeddingsLogger.error(
        EmbeddingComponent.HANDLER,
        'Failed to write error logs to file',
        context,
        { error: err.message },
      );
      embeddingsLogger.removeContext(requestId);
      return sendWithContext(
        req,
        sock,
        500,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Failed to write error logs', detail: err.message }),
      );
    }

    embeddingsLogger.info(
      EmbeddingComponent.HANDLER,
      `Received and logged ${processedLogs.length} error log(s)`,
      context,
      { count: processedLogs.length },
    );
    embeddingsLogger.removeContext(requestId);
    return sendWithContext(
      req,
      sock,
      201,
      { 'Content-Type': 'application/json' },
      JSON.stringify({
        status: 'received',
        message: 'Error logs processed successfully.',
        count: processedLogs.length,
      }),
    );
  },
};
