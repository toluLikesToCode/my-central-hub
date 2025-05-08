// filepath: /Users/toluadegbehingbe/my-central-hub/src/core/middlewares/cors.ts
import { Middleware } from '../router';
import logger from '../../utils/logger';
import { Socket } from 'net';
import { IncomingRequest } from '../../entities/http';
import { sendResponse } from '../../entities/sendResponse';
import { Readable } from 'stream';

/**
 * CORS middleware for adding CORS headers to all responses
 * Enables cross-origin requests from browsers to access the API
 */
export const corsMiddleware: Middleware = (req, sock, next) => {
  // Store the original sendResponse function reference
  const originalSendResponse = req.ctx?.sendResponse as typeof sendResponse | undefined;

  // Override sendResponse to add CORS headers
  req.ctx = req.ctx || {};
  req.ctx.sendResponse = (
    socket: Socket,
    statusCode: number,
    headers: Record<string, string>,
    body?: string | Buffer | Readable,
  ) => {
    // Add CORS headers to all responses
    const corsHeaders: Record<string, string> = {
      ...headers,
      'Access-Control-Allow-Origin': '*', // Allow all origins, can be restricted to specific origins
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With, X-Filename',
      'Access-Control-Expose-Headers':
        'Content-Length, Content-Range, ETag, Last-Modified, X-Total-Count, X-Total-Pages',
      'Access-Control-Max-Age': '86400', // Cache preflight requests for 24 hours
    };

    // Log CORS headers being added
    logger.debug('[corsMiddleware] Adding CORS headers to response', {
      requestId: req.ctx?.requestId,
      method: req.method,
      path: req.path,
      statusCode,
    });

    // Handle preflight OPTIONS requests
    if (req.method === 'OPTIONS') {
      return sendResponse(socket, 204, corsHeaders);
    }

    // Call the original sendResponse with added CORS headers
    return originalSendResponse
      ? originalSendResponse(socket, statusCode, corsHeaders, body)
      : sendResponse(socket, statusCode, corsHeaders, body);
  };

  return next();
};

/**
 * Handler for preflight OPTIONS requests
 * This can be used directly in route handlers for OPTIONS
 *
 * @param _req The incoming request (unused but kept for signature matching)
 * @param sock The client socket
 */
export const handlePreflightRequest = (_req: IncomingRequest, sock: Socket): void => {
  const headers: Record<string, string> = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With, X-Filename',
    'Access-Control-Expose-Headers':
      'Content-Length, Content-Range, ETag, Last-Modified, X-Total-Count, X-Total-Pages',
    'Access-Control-Max-Age': '86400', // Cache preflight requests for 24 hours
  };

  sendResponse(sock, 204, headers);
};
