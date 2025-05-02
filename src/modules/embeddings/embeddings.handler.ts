/* eslint-disable @typescript-eslint/no-explicit-any */
import { Socket } from 'net';
import { IncomingRequest } from '../../entities/http';
import { sendResponse } from '../../entities/sendResponse';
import {
  Logger,
  ConsoleTransport,
  FileTransport,
  JsonFormatter,
  PrettyFormatter,
} from '../../utils/logger';
import { embeddingService } from './embedding.service'; // Import the service
import path from 'path'; // Import path
import { config } from '../../config/server.config'; // Import config

const logger = new Logger({
  transports: [
    new ConsoleTransport({
      formatter: new PrettyFormatter(),
      level: 'info',
    }),
    new FileTransport({
      filename: 'logs/embeddings.log',
      formatter: new JsonFormatter(),
      level: 'debug',
    }),
  ],
  level: 'debug',
  exitOnError: false,
});

// Helper to summarize large arrays for logging
function summarizeArray(arr: any[]): string {
  if (!Array.isArray(arr)) return String(arr);
  const len = arr.length;
  if (len === 0) return '[]';
  const preview = arr.slice(0, 5).map((x) => Number(x).toFixed(4));
  let min = null,
    max = null;
  try {
    min = Math.min(...arr);
    max = Math.max(...arr);
  } catch {
    // empty catch to handle non-numeric arrays
  }
  return `[Array(len=${len}, min=${min}, max=${max}, preview=[${preview.join(', ')}]...)]`;
}

function summarizeObject(obj: any): any {
  if (Array.isArray(obj)) return summarizeArray(obj);
  if (obj && typeof obj === 'object') {
    const copy: any = Array.isArray(obj) ? [] : {};
    for (const key in obj) {
      if (Array.isArray(obj[key]) && obj[key].length > 20) {
        copy[key] = summarizeArray(obj[key]);
      } else if (typeof obj[key] === 'object' && obj[key] !== null) {
        copy[key] = summarizeObject(obj[key]);
      } else {
        copy[key] = obj[key];
      }
    }
    return copy;
  }
  return obj;
}

export const embeddingsController = {
  /**
   * Handles POST requests to /api/embeddings
   * Expects a JSON body: { "imagePaths": ["path1", "path2", ...] }
   * Returns JSON: { "path1": { "embedding": [...] }, "path2": { "embedding": [...] }, ... }
   */
  async handleEmbeddingsRequest(req: IncomingRequest, sock: Socket) {
    logger.info(
      `[EmbeddingsHandler] Received request: ` + JSON.stringify(summarizeObject(req.body)),
    );

    if (req.method !== 'POST') {
      return sendResponse(
        sock,
        405,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Method Not Allowed' }),
      );
    }

    if (!req.body || req.body.length === 0) {
      return sendResponse(
        sock,
        400,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Request body is empty' }),
      );
    }

    try {
      const body = JSON.parse(req.body.toString('utf-8'));
      const requestedPaths = body.imagePaths;

      if (!Array.isArray(requestedPaths) || requestedPaths.length === 0) {
        return sendResponse(
          sock,
          400,
          { 'Content-Type': 'application/json' },
          JSON.stringify({ error: 'No file paths provided.' }),
        );
      }

      // --- Security: Validate paths ---
      const mediaDir = path.resolve(config.mediaDir); // Get absolute configured media directory
      const imagePaths: string[] = []; // Store only validated paths
      for (const reqPath of requestedPaths) {
        if (typeof reqPath !== 'string') {
          logger.warn(`[EmbeddingsHandler] Invalid path type received: ${typeof reqPath}`);
          continue; // Skip non-string paths
        }
        const absoluteReqPath = path.resolve(mediaDir, reqPath);
        // Check for path traversal and ensure it's within the media directory
        if (!absoluteReqPath.startsWith(mediaDir) && !config.testMode) {
          logger.error(`[EmbeddingsHandler] Invalid path detected (outside media dir): ${reqPath}`);
          return sendResponse(
            sock,
            400,
            { 'Content-Type': 'application/json' },
            JSON.stringify({
              error: 'Invalid file path provided.',
              detail: `Path ${reqPath} is outside the allowed directory.`,
            }),
          );
        }
        imagePaths.push(absoluteReqPath); // Use the validated absolute path
      }

      if (imagePaths.length === 0) {
        logger.warn(`[EmbeddingsHandler] No valid paths remaining after validation.`);
        return sendResponse(
          sock,
          400,
          { 'Content-Type': 'application/json' },
          JSON.stringify({ error: 'No valid file paths provided.' }),
        );
      }
      // --- End Path Validation ---

      logger.info(
        `[EmbeddingsHandler] Requesting embeddings for ${imagePaths.length} validated paths.`,
      );
      const embeddingsResult = await embeddingService.getEmbeddings(imagePaths);
      logger.info(
        `[EmbeddingsHandler] Sending response for ${imagePaths.length} paths. Result summary: ` +
          JSON.stringify(summarizeObject(embeddingsResult)),
      );
      sendResponse(
        sock,
        200,
        { 'Content-Type': 'application/json' },
        JSON.stringify(embeddingsResult),
      );
    } catch (error: any) {
      logger.error(`[EmbeddingsHandler] Error processing embedding request: ${error.message}`);
      logger.error(`[EmbeddingsHandler] Error detail: ` + JSON.stringify(summarizeObject(error)));
      // Distinguish between JSON parsing errors and service errors
      const statusCode = error instanceof SyntaxError ? 400 : 500;
      const errorMessage =
        error instanceof SyntaxError
          ? 'Invalid JSON in request body.'
          : 'Failed to process embeddings.';
      sendResponse(
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
  async handleShutdownRequest(req: IncomingRequest, sock: Socket) {
    logger.info(`[EmbeddingsHandler] Received shutdown request`);
    if (req.method !== 'POST') {
      return sendResponse(
        sock,
        405,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Method Not Allowed' }),
      );
    }
    embeddingService.stop(); // Call stop
    logger.info(`[EmbeddingsHandler] Embedding service stop initiated.`);
    sendResponse(
      sock,
      200,
      { 'Content-Type': 'application/json' },
      JSON.stringify({ message: 'Embedding service shutdown initiated.' }),
    );
  },

  /**
   * Handles GET requests to /api/embeddings/status
   * Returns JSON: { state, isStarting, isProcessing, queueLength, currentBatch, error }
   */
  async handleStatusRequest(req: IncomingRequest, sock: Socket) {
    if (req.method !== 'GET') {
      return sendResponse(
        sock,
        405,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Method Not Allowed' }),
      );
    }
    const status = embeddingService.getStatus();
    sendResponse(sock, 200, { 'Content-Type': 'application/json' }, JSON.stringify(status));
  },
};
