/**
 * Router Module
 *
 * Implements a path-based router with middleware support for HTTP request handling.
 * The router matches incoming requests against registered routes and executes the
 * appropriate handler, with path parameter extraction.
 *
 * Features:
 * - Support for path parameters (/:id)
 * - Wildcard routes (*)
 * - Middleware chain execution
 * - Method-specific route registration (GET, POST, PUT, DELETE)
 * - Comprehensive request logging
 * - Error handling
 *
 * @module core/router
 */
import { Socket } from 'net';
import { IncomingRequest } from '../entities/http';
import { sendResponse } from '../entities/sendResponse';
import logger, { Logger } from '../utils/logger';

/* ───── Types ─────────────────────────────────────────────────────────── */

/**
 * Request handler function that processes an HTTP request
 */
export type Handler = (req: IncomingRequest, sock: Socket) => Promise<void> | void;

/**
 * Middleware function that can intercept requests before handlers
 * @param next - Function to continue the middleware chain
 */
export type Middleware = (
  req: IncomingRequest,
  sock: Socket,
  next: () => Promise<void>,
) => Promise<void> | void;

/**
 * Route configuration for request matching and handling
 */
interface Route {
  method: string; // 'GET' | 'POST' | 'ANY'
  regex: RegExp; // compiled path matcher
  keys: string[]; // param names (for :id stuff)
  handler: Handler;
  originalPath: string; // Store original path for better logging
}

/* ───── Router implementation ─────────────────────────────────────────── */

/**
 * Router class for HTTP request routing with middleware support
 */
class Router {
  private middlewares: Middleware[] = [];
  private routes: Route[] = [];

  /**
   * Register a middleware function to intercept all requests
   * @param mw - Middleware function to add
   */
  use(mw: Middleware) {
    this.middlewares.push(mw);
    logger.debug('Middleware registered', { middlewareCount: this.middlewares.length });
    return this; // Enable chaining
  }

  /**
   * Register a route with the specified method, path pattern, and handler
   * @param method - HTTP method (GET, POST, etc.) or 'ANY' for any method
   * @param path - URL path pattern with optional parameters
   * @param handler - Function to handle matching requests
   */
  add(method: string, path: string, handler: Handler) {
    const { regex, keys } = compilePath(path);
    this.routes.push({
      method: method.toUpperCase(),
      regex,
      keys,
      handler,
      originalPath: path,
    });
    logger.debug('Route registered', {
      method: method.toUpperCase(),
      path,
      paramKeys: keys,
    });
    return this; // Enable chaining
  }

  /**
   * Register a GET route
   */
  get(path: string, h: Handler) {
    return this.add('GET', path, h);
  }

  /**
   * Register a POST route
   */
  post(path: string, h: Handler) {
    return this.add('POST', path, h);
  }

  /**
   * Register a PUT route
   */
  put(path: string, h: Handler) {
    return this.add('PUT', path, h);
  }

  /**
   * Register a DELETE route
   */
  del(path: string, h: Handler) {
    return this.add('DELETE', path, h);
  }

  /**
   * Register a route that handles any HTTP method
   */
  any(path: string, h: Handler) {
    return this.add('ANY', path, h);
  }

  /**
   * Main request handler that processes incoming HTTP requests
   * @param req - Parsed incoming HTTP request
   * @param sock - Socket connection to client
   */
  async handle(req: IncomingRequest, sock: Socket): Promise<void> {
    const startTime = process.hrtime();
    const requestId = generateRequestId();

    // Create a request-scoped logger with request-specific metadata
    const reqLogger = logger.child({
      requestId,
      method: req.method,
      path: req.path,
      ip: sock.remoteAddress, // Get remote address from Socket instead of req
      userAgent: req.headers['user-agent'],
    });

    reqLogger.info(`Request started: ${req.method} ${req.path}`);

    try {
      // Handle OPTIONS requests for CORS support
      if (req.method === 'OPTIONS') {
        sendResponse(
          sock,
          200,
          {
            'Content-Type': 'text/plain',
            Allow: 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
          },
          'OK',
        );
        reqLogger.debug('Responded to OPTIONS request');
        logRequestCompletion(reqLogger, startTime, 200);
        return;
      }

      // Validate request path
      if (!req.path || typeof req.path !== 'string') {
        const status = 400;
        sendResponse(
          sock,
          status,
          {
            'Content-Type':
              req.path && req.path.startsWith('/api/') ? 'application/json' : 'text/plain',
          },
          req.path && req.path.startsWith('/api/')
            ? JSON.stringify({ error: 'Bad Request' })
            : 'Bad Request',
        );
        reqLogger.warn('Invalid request path', { path: req.path });
        logRequestCompletion(reqLogger, startTime, status);
        return;
      }

      // 1. run middleware chain
      let i = 0;
      // Use an explicit stack to unwind after handler
      const run = async (): Promise<void> => {
        if (i < this.middlewares.length) {
          const idx = i++;
          reqLogger.debug(`Executing middleware #${idx}`);
          await this.middlewares[idx](req, sock, run);
          return;
        }

        // Only after all middleware, run the handler
        // 2. route lookup
        const matching = this.routes.filter((r) => r.regex.test(req.path));
        reqLogger.debug('Route matching', {
          matchCount: matching.length,
          matchingRoutes: matching.map((r) => ({ method: r.method, path: r.originalPath })),
        });

        const route =
          matching.find((r) => r.method === req.method) ?? matching.find((r) => r.method === 'ANY');

        if (!route) {
          // Distinguish 404 vs 405
          if (matching.length) {
            // HTTP 405 Method Not Allowed
            const status = 405;
            // Only include allowed methods that are not 'ANY'
            const allowed = matching
              .map((r) => r.method)
              .filter((m) => m !== 'ANY')
              .join(', ');

            reqLogger.warn('Method not allowed', {
              allowedMethods: allowed,
              requestedMethod: req.method,
            });

            if (req.path.startsWith('/api/')) {
              sendResponse(
                sock,
                status,
                {
                  'Content-Type': 'application/json',
                  Allow: allowed,
                },
                JSON.stringify({ error: 'Method Not Allowed' }),
              );
            } else {
              sendResponse(sock, status, { Allow: allowed }, 'Method Not Allowed');
            }
            logRequestCompletion(reqLogger, startTime, status);
          } else {
            // HTTP 404 Not Found
            const status = 404;
            reqLogger.warn('No matching route found');

            if (req.path.startsWith('/api/')) {
              sendResponse(
                sock,
                status,
                { 'Content-Type': 'application/json' },
                JSON.stringify({ error: 'Not Found' }),
              );
            } else {
              sendResponse(sock, status, { 'Content-Type': 'text/plain' }, 'Not Found');
            }
            logRequestCompletion(reqLogger, startTime, status);
          }
          return;
        }

        // 3. pull params (/:id etc.) → req.ctx.params
        const match = route.regex.exec(req.path)!;
        const params: Record<string, string> = {};
        route.keys.forEach((k, idx) => {
          params[k] = decodeURIComponent(match[idx + 1]);
        });
        (req.ctx ??= {}).params = params;

        reqLogger.debug('Route matched', {
          method: route.method,
          path: route.originalPath,
          params,
        });

        // 4. invoke handler
        try {
          reqLogger.debug('Executing route handler');
          // Store status in req.ctx for logging
          (req.ctx ??= {}).responseStatus = 200; // Default if not set by handler

          await route.handler(req, sock);

          // Log after handler completes
          logRequestCompletion(reqLogger, startTime, req.ctx.responseStatus || 200);
        } catch (err) {
          const status = 500;
          const error = err as Error;

          reqLogger.error('Handler error', {
            error: {
              message: error.message,
              stack: error.stack,
              name: error.name,
            },
          });

          if (req.path.startsWith('/api/')) {
            sendResponse(
              sock,
              status,
              { 'Content-Type': 'application/json' },
              JSON.stringify({ error: 'Internal Server Error' }),
            );
          } else {
            sendResponse(sock, status, { 'Content-Type': 'text/plain' }, '500 Server Error');
          }

          logRequestCompletion(reqLogger, startTime, status);
        }
      };

      await run();
    } catch (err) {
      // Global error handler for middleware
      const status = 500;
      const error = err as Error;

      reqLogger.error('Middleware error', {
        error: {
          message: error.message,
          stack: error.stack,
          name: error.name,
        },
      });

      if (req.path.startsWith('/api/')) {
        sendResponse(
          sock,
          status,
          { 'Content-Type': 'application/json' },
          JSON.stringify({ error: 'Internal Server Error' }),
        );
      } else {
        sendResponse(sock, status, { 'Content-Type': 'text/plain' }, '500 Server Error');
      }

      logRequestCompletion(reqLogger, startTime, status);
    }
  }
}

/**
 * Generate a unique request ID for tracking requests in logs
 */
function generateRequestId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substring(2);
}

/**
 * Log the completion of a request with duration and status
 */
function logRequestCompletion(
  logger: Logger,
  startTime: [number, number],
  status: number | unknown,
): void {
  const [seconds, nanoseconds] = process.hrtime(startTime);
  const duration = seconds * 1000 + nanoseconds / 1000000; // convert to ms

  // Ensure status is a number
  const statusCode = typeof status === 'number' ? status : 500;

  // Determine log level based on status code
  if (statusCode >= 500) {
    logger.error(`Request completed with status ${statusCode}`, {
      status: statusCode,
      durationMs: duration,
    });
  } else if (statusCode >= 400) {
    logger.warn(`Request completed with status ${statusCode}`, {
      status: statusCode,
      durationMs: duration,
    });
  } else {
    logger.info(`Request completed with status ${statusCode}`, {
      status: statusCode,
      durationMs: duration,
    });
  }
}

/* ───── Path pattern compiler ───────────────────────────────────────────
   /files/:name  ->  ^/files/([^/]+)$           keys=['name']
   /api/*        ->  ^/api/(.*)$                keys=['*']
------------------------------------------------------------------------ */

/**
 * Compile a path pattern into a regular expression for route matching
 * @param pattern - URL path pattern with optional parameters
 * @returns Object with compiled regex and extracted parameter keys
 *
 * @example
 * compilePath('/users/:id') // { regex: /^\/users\/([^/]+)$/, keys: ['id'] }
 * compilePath('/files/*')   // { regex: /^\/files\/(.*)$/, keys: ['*'] }
 */
function compilePath(pattern: string): { regex: RegExp; keys: string[] } {
  const keys: string[] = [];
  const regexSrc = pattern
    .replace(/\/:(\w+)/g, (_, k) => {
      keys.push(k);
      return '/([^/]+)';
    })
    .replace(/\*/g, () => {
      keys.push('*');
      return '(.*)';
    });
  return { regex: new RegExp(`^${regexSrc}$`), keys };
}

/* ───── Exports ───────────────────────────────────────────────────────── */

/**
 * Create a new router instance
 * @returns A new Router instance
 */
export function createRouter(): Router {
  logger.info('Creating new router instance');
  return new Router();
}

// Export a singleton router instance
export default createRouter();
