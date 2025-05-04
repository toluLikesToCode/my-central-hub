/**
 * src/modules/embeddings/embeddings.handler.ts
 * This file handles the HTTP requests for the embeddings service.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { Socket } from 'net';
import { IncomingRequest } from '../../entities/http';
import { sendResponse } from '../../entities/sendResponse';
import { embeddingService } from './embedding.service';
import { embeddingsLogger, EmbeddingComponent } from './embeddingsLogger';
import path from 'path';
import { config } from '../../config/server.config';
import { randomUUID } from 'crypto';

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
    const requestId = randomUUID();
    const context = embeddingsLogger.createContext({
      requestId,
      source: 'http-request',
    });

    embeddingsLogger.info(EmbeddingComponent.HANDLER, 'Received embedding request', context, {
      bodyLength: req.body?.length || 0,
    });

    if (req.method !== 'POST') {
      embeddingsLogger.warn(
        EmbeddingComponent.HANDLER,
        `Method not allowed: ${req.method}`,
        context,
        { method: req.method },
      );

      embeddingsLogger.removeContext(requestId);
      return sendResponse(
        sock,
        405,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Method Not Allowed, use POST' }),
      );
    }

    if (!req.body || req.body.length === 0) {
      embeddingsLogger.warn(EmbeddingComponent.HANDLER, 'Request body is empty', context);

      embeddingsLogger.removeContext(requestId);
      return sendResponse(
        sock,
        400,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Request body is empty' }),
      );
    }

    try {
      const body = JSON.parse(req.body.toString('utf-8'));
      let requestedPaths = body.imagePaths;

      embeddingsLogger.debug(EmbeddingComponent.HANDLER, 'Parsed request body', context, {
        body: summarizeObject(body),
        pathsRequested: Array.isArray(requestedPaths)
          ? requestedPaths.length
          : typeof requestedPaths === 'string'
            ? 1
            : 0,
      });

      // Preserve original client-provided paths for remapping in response
      const rawPaths: string[] = Array.isArray(requestedPaths)
        ? [...requestedPaths]
        : [requestedPaths];

      // sanitize the requested paths
      if (typeof requestedPaths === 'string') {
        // If a single string is provided, convert it to an array
        requestedPaths = [requestedPaths];
      } else if (Array.isArray(requestedPaths)) {
        // If an array is provided, ensure all elements are strings
        requestedPaths = requestedPaths.filter((path) => typeof path === 'string');
      } else {
        // If neither, return an error
        embeddingsLogger.warn(
          EmbeddingComponent.HANDLER,
          'Invalid imagePaths format, expected an array of strings',
          context,
          { imagePaths: requestedPaths },
        );

        embeddingsLogger.removeContext(requestId);
        return sendResponse(
          sock,
          400,
          { 'Content-Type': 'application/json' },
          JSON.stringify({ error: 'Invalid imagePaths format. Expected an array of strings.' }),
        );
      }

      // normalize the paths
      // Normalize the requested paths: trim whitespace, collapse redundant segments,
      // strip any leading slashes or parent‑directory references to keep them relative
      requestedPaths = requestedPaths
        .map((p: string): string => p.trim())
        .map(
          (p: string): string =>
            path
              .normalize(p)
              .replace(/^(\.\.[/\\])+/, '') // remove "../" segments
              .replace(/^[/\\]+/, ''), // remove leading "/" or "\"
        );

      if (!Array.isArray(requestedPaths) || requestedPaths.length === 0) {
        embeddingsLogger.warn(
          EmbeddingComponent.HANDLER,
          'No file paths provided after normalization',
          context,
        );

        embeddingsLogger.removeContext(requestId);
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

      embeddingsLogger.debug(
        EmbeddingComponent.HANDLER,
        `Validating ${requestedPaths.length} paths`,
        context,
        { mediaDir },
      );

      for (const reqPath of requestedPaths) {
        if (typeof reqPath !== 'string') {
          embeddingsLogger.warn(
            EmbeddingComponent.HANDLER,
            `Invalid path type received: ${typeof reqPath}`,
            context,
            { pathType: typeof reqPath },
          );
          continue; // Skip non-string paths
        }

        const absoluteReqPath = path.resolve(mediaDir, reqPath);
        // Check for path traversal and ensure it's within the media directory
        if (!absoluteReqPath.startsWith(mediaDir) && !config.testMode) {
          embeddingsLogger.error(
            EmbeddingComponent.HANDLER,
            `Invalid path detected (outside media dir): ${reqPath}`,
            context,
            {
              requestedPath: reqPath,
              absolutePath: absoluteReqPath,
              mediaDir,
            },
          );

          embeddingsLogger.removeContext(requestId);
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
        embeddingsLogger.warn(
          EmbeddingComponent.HANDLER,
          'No valid paths remaining after validation.',
          context,
        );

        embeddingsLogger.removeContext(requestId);
        return sendResponse(
          sock,
          400,
          { 'Content-Type': 'application/json' },
          JSON.stringify({ error: 'No valid file paths provided.' }),
        );
      }
      // --- End Path Validation ---

      embeddingsLogger.info(
        EmbeddingComponent.HANDLER,
        `Requesting embeddings for ${imagePaths.length} validated paths.`,
        context,
        {
          validatedPathCount: imagePaths.length,
          mediaDir,
        },
      );

      // Update context with media count
      embeddingsLogger.updateContext(requestId, {
        mediaCount: imagePaths.length,
      });

      const embeddingsResult = await embeddingService.getEmbeddings(imagePaths, rawPaths);

      embeddingsLogger.info(
        EmbeddingComponent.HANDLER,
        `Sending response for ${imagePaths.length} paths`,
        context,
        {
          pathCount: imagePaths.length,
          hasErrors: Object.values(embeddingsResult).some((entry) => !!entry.error),
        },
      );

      embeddingsLogger.removeContext(requestId);
      sendResponse(
        sock,
        200,
        { 'Content-Type': 'application/json' },
        JSON.stringify(embeddingsResult),
      );
    } catch (error: any) {
      embeddingsLogger.error(
        EmbeddingComponent.HANDLER,
        `Error processing embedding request: ${error.message}`,
        context,
        { error },
      );

      // Distinguish between JSON parsing errors and service errors
      const statusCode = error instanceof SyntaxError ? 400 : 500;
      const errorMessage =
        error instanceof SyntaxError
          ? 'Invalid JSON in request body.'
          : 'Failed to process embeddings.';

      embeddingsLogger.removeContext(requestId);
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
    const requestId = randomUUID();
    const context = embeddingsLogger.createContext({
      requestId,
      source: 'shutdown-request',
    });

    embeddingsLogger.info(EmbeddingComponent.HANDLER, 'Received shutdown request', context);

    embeddingService.stop();

    embeddingsLogger.info(EmbeddingComponent.HANDLER, 'Embedding service stop initiated.', context);

    embeddingsLogger.removeContext(requestId);
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
    const requestId = randomUUID();
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
      return sendResponse(
        sock,
        405,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Method Not Allowed' }),
      );
    }

    // If service is stopped or in error, start Python process on status ping
    let status = embeddingService.getStatus();
    if ((status.state === 'STOPPED' || status.state === 'ERROR') && !status.isStarting) {
      embeddingsLogger.info(
        EmbeddingComponent.HANDLER,
        'Status ping received: starting Python process.',
        context,
      );

      embeddingService.startPythonProcess().catch((err) => {
        embeddingsLogger.error(
          EmbeddingComponent.HANDLER,
          `Failed to start Python process on status ping: ${err.message}`,
          context,
          { error: err },
        );
      });

      // Refresh status after initiating start
      status = embeddingService.getStatus();
    }

    embeddingsLogger.debug(
      EmbeddingComponent.HANDLER,
      `Returning status: ${status.state}`,
      context,
      status,
    );

    embeddingsLogger.removeContext(requestId);
    sendResponse(sock, 200, { 'Content-Type': 'application/json' }, JSON.stringify(status));
  },
};
