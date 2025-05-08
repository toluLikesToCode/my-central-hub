/**
 * src/entities/sendResponse.ts
 * This file handles the HTTP response sending logic.
 */
import { Socket } from 'net';
import { Readable } from 'stream';
import logger from '../utils/logger';

const STATUS_TEXT: Record<number, string> = {
  200: 'OK',
  201: 'Created', // Added for common use
  204: 'No Content', // Added for common use
  206: 'Partial Content',
  301: 'Moved Permanently', // Added
  302: 'Found', // Added (often used for temporary redirect)
  304: 'Not Modified', // Added
  400: 'Bad Request',
  401: 'Unauthorized', // Added
  403: 'Forbidden', // Added
  404: 'Not Found',
  405: 'Method Not Allowed',
  416: 'Range Not Satisfiable',
  500: 'Internal Server Error',
  502: 'Bad Gateway', // Added
  503: 'Service Unavailable', // Added
};

// Add a new interface for chunked response options
export interface ChunkedResponseOptions {
  status: number;
  headers: Record<string, string>;
  requestId?: string;
  noBuffering?: boolean; // Set to true to prevent proxy buffering
}

export function sendResponse(
  socket: Socket,
  status: number,
  // User-provided headers. This function might modify a copy.
  initialHeaders: Record<string, string>,
  body?: string | Buffer | Readable,
): void {
  if (socket.destroyed) {
    logger.debug('[sendResponse] Attempted to write to destroyed socket', { status });
    return;
  }

  try {
    const finalHeaders = { ...initialHeaders }; // Work with a copy

    // Automatically set Content-Length for string or Buffer bodies
    // if not already set by the caller and not using chunked encoding.
    // Inside sendResponse in sendResponse.ts
    if (
      body &&
      (typeof body === 'string' || Buffer.isBuffer(body)) &&
      !finalHeaders['Content-Length'] &&
      finalHeaders['Transfer-Encoding']?.toLowerCase() !== 'chunked'
    ) {
      const len = Buffer.byteLength(body).toString();
      finalHeaders['Content-Length'] = len;
      logger.debug(
        `[sendResponse] Automatically set Content-Length: ${len} for path: ${(socket as Socket & { _httpMessage?: { url?: string } })._httpMessage?.url || initialHeaders['X-Original-URL'] || 'unknown'}`,
      );
    }

    // Automatically set a default Content-Type if a body is present and no Content-Type is specified.
    // Route handlers should ideally set a more specific Content-Type.
    if (body && !finalHeaders['Content-Type']) {
      logger.warn(
        '[sendResponse] Content-Type not set by handler, defaulting to application/octet-stream.',
        { status, path: 'unknown' }, // Path not available on net.Socket; set to 'unknown'
      );
      finalHeaders['Content-Type'] = 'application/octet-stream';
    }

    // Determine if the connection should be closed based on final headers
    const shouldCloseConnection = Object.entries(finalHeaders).some(
      ([key, value]) => key.toLowerCase() === 'connection' && value.toLowerCase() === 'close',
    );

    // Determine if content is binary based on final Content-Type
    const contentTypeHeader = finalHeaders['Content-Type'] || '';
    const isBinaryContent =
      contentTypeHeader.includes('application/octet-stream') ||
      contentTypeHeader.includes('image/') ||
      contentTypeHeader.includes('video/') ||
      contentTypeHeader.includes('audio/');

    // Prepare HTTP status line and headers string
    const headerLines = Object.entries(finalHeaders)
      .map(([k, v]) => `${k}: ${v}`)
      .join('\r\n');
    const head = `HTTP/1.1 ${status} ${STATUS_TEXT[status] ?? 'Status'}\r\n${headerLines}\r\n\r\n`;

    // Send headers
    socket.write(head);

    // Handle responses with no body
    if (!body) {
      if (shouldCloseConnection && !socket.destroyed) {
        socket.end();
      }
      return;
    }

    // Handle Readable stream bodies
    if (body instanceof Readable) {
      logger.debug('[sendResponse] Piping stream body', {
        status,
        isBinary: isBinaryContent,
        contentType: finalHeaders['Content-Type'],
        shouldClose: shouldCloseConnection,
      });

      // For streams, Content-Length must be set by the caller if known,
      // or Transfer-Encoding: chunked should be used (via beginChunkedResponse).
      // If neither is true for a keep-alive connection, the client might hang.

      body.on('error', (err) => {
        logger.error('[sendResponse] Stream body error caught', {
          error: err.message,
          status,
          socketDestroyed: socket.destroyed,
        });
        if (!socket.destroyed) {
          // Abruptly end on stream error to prevent hanging or partial data issues.
          socket.destroy(err); // Using destroy might be safer than end for error cases
        }
      });

      body.on('end', () => {
        logger.debug('[sendResponse] Stream body ended.', {
          isBinary: isBinaryContent,
          shouldClose: shouldCloseConnection,
        });
        if (shouldCloseConnection && !socket.destroyed) {
          // Ensure data is flushed before closing for stream an 'end'
          // For non-chunked streams, this 'end' is crucial if Content-Length was set.
          // If it was keep-alive without proper framing, client might still be waiting.
          socket.end();
        }
        // If not shouldCloseConnection, the socket remains open.
        // Client relies on Content-Length (if set for the stream) or client-side timeout.
      });

      // Pipe the stream.
      // If shouldCloseConnection is true, pipe should end the socket.
      // If shouldCloseConnection is false (keep-alive):
      //   - And Content-Length was set for the stream: client reads N bytes.
      //   - And no Content-Length/chunking: client will hang. This function assumes
      //     proper framing (Content-Length or chunking via beginChunkedResponse) for keep-alive streams.
      body.pipe(socket, { end: shouldCloseConnection });

      return;
    }

    // Handle string or Buffer bodies
    // (Content-Length was auto-calculated above if not provided by caller)
    logger.debug(`[sendResponse] Writing ${typeof body} body`, {
      status,
      size: finalHeaders['Content-Length'],
      contentType: finalHeaders['Content-Type'],
      shouldClose: shouldCloseConnection,
    });

    socket.write(body, (err?: Error) => {
      if (err && !socket.destroyed) {
        logger.error('[sendResponse] Error writing string/buffer body', {
          error: err.message,
          status,
        });
        socket.destroy(err); // Destroy on write error
        return;
      }
      if (shouldCloseConnection && !socket.destroyed) {
        socket.end();
      }
      // If not shouldCloseConnection, socket remains open. Client relies on Content-Length.
    });
  } catch (err) {
    logger.error('[sendResponse] General error during response sending', {
      error: (err as Error).message,
      status,
      socketDestroyed: socket.destroyed,
    });
    if (!socket.destroyed) {
      // Fallback: try to end the socket if any error occurs and it's not already closed.
      socket.destroy(err as Error);
    }
  }
}

/**
 * Begins a chunked HTTP response, allowing for progressive data streaming.
 * This is useful for long-running operations to prevent socket timeouts.
 *
 * @param socket The client socket connection
 * @param options Response options including status and headers
 * @returns A function that can be used to send chunks
 */
export function beginChunkedResponse(
  socket: Socket,
  options: ChunkedResponseOptions,
): {
  sendChunk: (data: unknown) => boolean; // Returns false if socket is destroyed or write fails
  endResponse: () => void;
} {
  if (socket.destroyed) {
    logger.debug('[beginChunkedResponse] Attempted to use destroyed socket', {
      status: options.status,
    });
    return {
      sendChunk: () => false,
      endResponse: () => {},
    };
  }

  try {
    const finalHeaders: Record<string, string> = {
      'Transfer-Encoding': 'chunked', // Essential for chunked responses
      Connection: 'keep-alive', // Chunked responses are typically used with keep-alive
      'Cache-Control': 'no-cache, no-store, must-revalidate', // Good for dynamic content
      ...options.headers, // Allow user to override or add headers
    };

    if (options.requestId) {
      finalHeaders['X-Request-ID'] = options.requestId;
    }
    if (options.noBuffering) {
      finalHeaders['X-Accel-Buffering'] = 'no';
    }

    const headerLines = Object.entries(finalHeaders)
      .map(([k, v]) => `${k}: ${v}`)
      .join('\r\n');
    const head = `HTTP/1.1 ${options.status} ${STATUS_TEXT[options.status] ?? 'Status'}\r\n${headerLines}\r\n\r\n`;

    socket.write(head);

    return {
      sendChunk: (data: unknown): boolean => {
        if (socket.destroyed) {
          logger.warn('[beginChunkedResponse] Attempted to send chunk on destroyed socket');
          return false;
        }
        try {
          const chunkData = typeof data === 'string' ? data : JSON.stringify(data);
          const chunkLengthHex = Buffer.byteLength(chunkData).toString(16);
          // Returns true if the entire data was flushed successfully to the kernel buffer.
          // Returns false if all or part of the data was queued in user memory. 'drain' will be emitted when the buffer is again free.
          return socket.write(`${chunkLengthHex}\r\n${chunkData}\r\n`);
        } catch (chunkErr) {
          logger.error('[beginChunkedResponse] Error writing chunk', {
            error: (chunkErr as Error).message,
          });
          if (!socket.destroyed) socket.destroy(chunkErr as Error);
          return false;
        }
      },
      endResponse: () => {
        if (socket.destroyed) {
          logger.warn('[beginChunkedResponse] Attempted to end response on destroyed socket');
          return;
        }
        try {
          socket.write('0\r\n\r\n'); // Final chunk
          // For chunked encoding with Connection: keep-alive, the client knows the response ends
          // with the 0-length chunk. The socket should remain open for subsequent requests.
          // If Connection: close was somehow set, then socket.end() would be appropriate.
          // Here, we assume typical keep-alive for chunked.
          if (finalHeaders['Connection']?.toLowerCase() === 'close' && !socket.destroyed) {
            socket.end();
          }
        } catch (endErr) {
          logger.error('[beginChunkedResponse] Error writing final chunk', {
            error: (endErr as Error).message,
          });
          if (!socket.destroyed) socket.destroy(endErr as Error);
        }
      },
    };
  } catch (err) {
    logger.error('[beginChunkedResponse] Error initializing chunked response', {
      error: (err as Error).message,
      status: options.status,
      socketDestroyed: socket.destroyed,
    });
    if (!socket.destroyed) {
      socket.destroy(err as Error);
    }
    return {
      sendChunk: () => false,
      endResponse: () => {},
    };
  }
}

// Important Notes on this Updated Code:

// Status Text Expansion: I added a few more common HTTP status codes to STATUS_TEXT for completeness.
// Error Handling in sendResponse: Using socket.destroy(err) in error paths within sendResponse can be more robust than socket.end() as it ensures immediate closure and signals an error state.
// Stream Piping in sendResponse: The line body.pipe(socket, { end: shouldCloseConnection }); is a simplified way to handle stream ending. For robust keep-alive streaming where Content-Length isn't known, beginChunkedResponse is the preferred method. If sendResponse is used with a stream and keep-alive, the caller must ensure Content-Length is provided in initialHeaders, or the client might hang.
// beginChunkedResponse Return Value: sendChunk now returns a boolean (like socket.write) indicating if the data was flushed or buffered, which can be useful for flow control if sending large/fast chunks.
// Idempotency and Safety: socket.destroyed checks are crucial before any socket operation.
// Logging Context: Added a bit more context to some log messages.
// Charset in Content-Type: It's good practice to include charset=utf-8 for text-based content types like application/json. While not added automatically here (to keep the function focused), your handlers should do this: {'Content-Type': 'application/json; charset=utf-8'}.

// Stream Handling Clarification:

// The existing stream handling logic is largely preserved. For Readable streams, Content-Length is not automatically calculated by this function because the length is generally unknown upfront for a generic stream.
// If you are sending streams and want keep-alive, you should either:
// Use beginChunkedResponse (which correctly uses Transfer-Encoding: chunked).
// Ensure the caller of sendResponse provides an accurate Content-Length header if the stream's total length is known beforehand.
// Or ensure Connection: close is set if proper framing for keep-alive (via Content-Length or chunking) isn't possible for the stream.
// The body.pipe(socket, { end: false }) for non-binary streams is tricky for keep-alive without Content-Length or chunking. I've added comments and a slight modification to use { end: shouldCloseConnection } for body.pipe as a general approach, but for robust keep-alive streaming, beginChunkedResponse is better.
