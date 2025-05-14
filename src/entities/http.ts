import { Socket } from 'net';

/**
 * src/entities/http.ts
 * This file contains the interface definitions for HTTP requests.
 */

export interface SecurityContext {
  /**
   * Indicates if the request has been authenticated
   */
  authenticated?: boolean;

  /**
   * Client IP address for security logging and rate limiting
   */
  clientIp?: string;

  /**
   * Security validation timestamp
   */
  validatedAt?: Date;

  /**
   * Request origin for CORS validation
   */
  origin?: string;

  /**
   * Authentication token or session ID if available
   */
  token?: string;

  /**
   * Custom security flags and values
   */
  flags?: Record<string, boolean>;

  /**
   * Rate limiting data
   */
  rateLimit?: {
    remaining: number;
    limit: number;
    reset: number;
  };
}

export interface RequestContext {
  /**
   * Optional custom sendResponse function for this request (used by router/middleware)
   */
  sendResponse?: (
    sock: Socket,
    status: number,
    headers: Record<string, string>,
    body?: string | Buffer | import('stream').Readable,
  ) => void;
  // Allow any other fields for extensibility
  [key: string]: unknown;
}

export interface IncomingRequest {
  url: URL; // canonical URL (always present)
  path: string; // == url.pathname
  query: Record<string, string>; // decoded single-value map
  httpVersion: string; // e.g. "HTTP/1.1"

  method: string;
  headers: Record<string, string>;
  headersMap?: Map<string, string[]>;
  body?: Buffer;
  raw: string;
  // ctx is a generic object for passing data between middlewares
  // and handlers. It can be used to store request-specific data.
  // data can be accessed by middlewares and handlers
  // ctx.requestId = req.ctx.requestId || generateRequestId();
  // the return type of req.ctx.requestId is {} which is a generic object
  // that converted to a string by calling toString() on it.

  ctx?: RequestContext;

  invalid?: boolean;

  /**
   * Security context for tracking authentication status,
   * rate limiting, and other security features
   */
  security?: SecurityContext;

  /**
   * Request validation errors
   */
  validationErrors?: string[];

  /**
   * Request timing information (for monitoring/performance tracking)
   */
  timing?: {
    startedAt: number;
    parsedAt?: number;
    routedAt?: number;
    respondedAt?: number;
  };
}
