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
import { sendWithContext } from '../entities/sendResponse';
import logger, { Logger } from '../utils/logger';
import { requestIdMiddleware } from '../core/middlewares/requestId';
import { corsMiddleware } from '../core/middlewares/cors';
import { optionsHandlerMiddleware } from '../core/middlewares/optionsHandler';
//import { stringify } from 'node:querystring';
// import { stringify } from 'node:querystring'; use this later if needed
// here is a short example of how to use it:
/*
Example usage of querystring.stringify:

const params = { foo: 'bar', baz: 'qux' };
const queryString = stringify(params); // "foo=bar&baz=qux"
*/

// more below

// ```typescript
// import { IncomingRequest } from '../../entities/http';
// import { Socket } from 'net';
// import { sendResponse } from '../../entities/sendResponse';
// import logger from '../../utils/logger';

// /**
//  * Authentication middleware that validates JWT tokens in the Authorization header
//  */
// export const authMiddleware: Middleware = async (
//   req: IncomingRequest,
//   sock: Socket,
//   next: () => Promise<void>
// ) => {
//   const authHeader = req.headers['authorization'];

//   // Skip auth check for public routes
//   if (req.path.startsWith('/public/') || req.path === '/login') {
//     return await next();
//   }

//   if (!authHeader || !authHeader.startsWith('Bearer ')) {
//     logger.warn('Authentication failed: Missing or invalid Authorization header');
//     sendResponse(
//       sock,
//       401,
//       { 'Content-Type': 'application/json' },
//       JSON.stringify({ error: 'Unauthorized: Missing or invalid token' })
//     );
//     return;
//   }

//   const token = authHeader.split(' ')[1];

//   try {
//     // In a real app, you would validate the token here
//     // const userData = verifyJwtToken(token);
//     const userData = { id: '123', username: 'exampleUser', role: 'admin' };

//     // Set user data in request context for handlers to use
//     (req.ctx ??= {}).user = userData;

//     // Log successful authentication
//     logger.debug('User authenticated', { userId: userData.id });

//     // Continue to the next middleware or route handler
//     await next();
//   } catch (error) {
//     logger.error('Token validation failed', { error });
//     sendResponse(
//       sock,
//       401,
//       { 'Content-Type': 'application/json' },
//       JSON.stringify({ error: 'Unauthorized: Invalid token' })
//     );
//   }
// };
// ```

// Now you can register this middleware with your router:

// ```typescript
// import router from './core/router';
// import { authMiddleware } from './core/middlewares/auth';

// // Register the auth middleware - will run on all requests
// router.use(authMiddleware);

// // Protected route that requires authentication
// router.get('/api/profile', async (req, sock) => {
//   // The user data is now available in req.ctx.user
//   const userData = req.ctx?.user;

//   sendResponse(
//     sock,
//     200,
//     { 'Content-Type': 'application/json' },
//     JSON.stringify({
//       profile: userData,
//       message: 'Profile accessed successfully'
//     })
//   );
// });
// ```

// This example demonstrates how the middleware:
// 1. Intercepts each request
// 2. Checks for authentication
// 3. Either rejects unauthorized requests or enriches the request context with user data
// 4. Passes control to the next middleware or handler using the `next()` function

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
    logger.debug('Middleware registered', { middlewareCount: this.middlewares.length, mw });
    return this; // Enable chaining
  }

  /**
   * Reset the router state (useful for testing)
   * Clears all registered routes and middlewares
   */
  reset(): this {
    this.routes = [];
    this.middlewares = [];
    logger.debug('Router state reset');
    return this; // Enable chaining
  }

  /**
   * Normalize a path: ensures leading slash, removes trailing slash (except for root)
   */
  private static normalizePath(path: string): string {
    if (!path.startsWith('/')) path = '/' + path;
    if (path.length > 1 && path.endsWith('/')) path = path.replace(/\/+$/, '');
    return path;
  }

  /**
   * Register a route with the specified method, path pattern, and handler
   * @param method - HTTP method (GET, POST, etc.) or 'ANY' for any method
   * @param path - URL path pattern with optional parameters
   * @param handler - Function to handle matching requests
   */
  add(method: string, path: string, handler: Handler) {
    const normalizedPath = Router.normalizePath(path);
    const { regex, keys } = compilePath(normalizedPath);
    this.routes.push({
      method: method.toUpperCase(),
      regex,
      keys,
      handler,
      originalPath: normalizedPath,
    });
    logger.debug('Route registered', {
      method: method.toUpperCase(),
      path: normalizedPath,
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
   * Register a HEAD route
   */
  head(path: string, h: Handler) {
    return this.add('HEAD', path, h);
  }

  /**
   * Register an OPTIONS route
   */
  options(path: string, h: Handler) {
    return this.add('OPTIONS', path, h);
  }

  /**
   * Register a route that handles any HTTP method
   */
  any(path: string, h: Handler) {
    return this.add('ANY', path, h);
  }

  /**
   * Register a debug endpoint that lists all registered routes
   */
  registerDebugRoute() {
    this.get('/__routes__', (req, sock) => {
      const routesList = this.routes.map((r) => ({
        method: r.method,
        path: r.originalPath,
      }));
      sendWithContext(
        req,
        sock,
        200,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ routes: routesList }, null, 2),
      );
    });
    return this;
  }

  /**
   * Helper to capture headers from a handler without sending a body
   * Simulates Express's approach for HEAD requests
   */
  private async captureHeadersFromHandler(
    handler: Handler,
    req: IncomingRequest,
    sock: Socket,
  ): Promise<{ status: number; headers: Record<string, string> }> {
    let capturedStatus = 200;
    let capturedHeaders: Record<string, string> = {};
    // Save the original sendResponse from context (if any)
    const originalSendResponse = req.ctx?.sendResponse;
    // Inject a capturing sendResponse into the request context
    req.ctx = req.ctx || {};
    req.ctx.sendResponse = (_sock: Socket, status: number, headers: Record<string, string>) => {
      capturedStatus = status;
      capturedHeaders = { ...headers };
    };
    try {
      await handler(req, sock);
    } finally {
      // Restore the original sendResponse in context
      if (originalSendResponse) {
        req.ctx.sendResponse = originalSendResponse;
      } else {
        delete req.ctx.sendResponse;
      }
    }
    return { status: capturedStatus, headers: capturedHeaders };
  }

  /**
   * Main request handler that processes incoming HTTP requests
   * @param req - Parsed incoming HTTP request
   * @param sock - Socket connection to client
   */
  async handle(req: IncomingRequest, sock: Socket): Promise<void> {
    const startTime = process.hrtime();
    // Normalize path to always start with a slash and remove trailing slash (except root)
    if (req.path) {
      req.path = Router.normalizePath(req.path);
    }
    let requestId = req.ctx?.requestId || req.headers['x-request-id'] || req.headers['request-id'];
    if (!requestId) {
      const newRequestId = crypto.randomUUID();
      req.ctx = { ...(req.ctx || {}), requestId: newRequestId };
      requestId = newRequestId;
      req.headers['x-request-id'] = newRequestId;
      req.headers['request-id'] = newRequestId;
      logger.warn('Generated new request ID', { requestId });
    }
    const reqLogger = logger.child({
      requestId,
      path: req.path,
      ip: sock.remoteAddress,
      headers: req.headers,
      body: req.body?.byteLength || 0,
      ctx: req.ctx,
    });
    reqLogger.info(`Request started: ${req.method} ${req.path}`);
    try {
      // Validate request path
      if (!req.path || typeof req.path !== 'string') {
        const status = 400;
        const isApi = req.path?.startsWith('/api/');
        sendWithContext(
          req,
          sock,
          status,
          { 'Content-Type': isApi ? 'application/json' : 'text/plain' },
          isApi ? JSON.stringify({ error: 'Bad Request' }) : 'Bad Request',
        );
        reqLogger.warn('Invalid request path', { path: req.path });
        logRequestCompletion(reqLogger, startTime, status);
        return;
      }

      // Special handling for OPTIONS requests (for CORS preflight)
      if (req.method === 'OPTIONS') {
        reqLogger.debug('Handling OPTIONS request');

        // Continue to middleware chain for OPTIONS requests
        // This allows our registerd CORS middleware and OPTIONS handlers to process the request
        // We're not ending the request here to support tests that check specific headers
      }
      // run middlewear chain
      await this.runMiddlewares(req, sock, reqLogger);
      const matching = this.routes.filter((r) => {
        const m = req.path.match(r.regex);
        if (!m) return false;
        if (r.keys.length) {
          req.ctx = { ...(req.ctx || {}), params: {} };
          r.keys.forEach((k, i) => {
            (req.ctx!.params as Record<string, string>)[k] = m[i + 1];
          });
        }
        return true;
      });
      // If no routes match the path
      if (matching.length === 0) {
        reqLogger.debug('No routes match request path');
        // Find similar routes for suggestion
        const availablePaths = this.routes.map((r) => r.originalPath);
        const suggestion =
          availablePaths.length > 0
            ? `Available routes: ${availablePaths.join(', ')}`
            : 'No routes registered.';
        const isApi = req.path.startsWith('/api/');
        if (isApi) {
          sendWithContext(
            req,
            sock,
            404,
            { 'Content-Type': 'application/json' },
            JSON.stringify({
              error: 'Not Found',
              message: `No route matches ${req.path}`,
              suggestion,
            }),
          );
        } else {
          sendWithContext(
            req,
            sock,
            404,
            { 'Content-Type': 'text/plain' },
            `404 Not Found: ${req.path}\n${suggestion}`,
          );
        }
        logRequestCompletion(reqLogger, startTime, 404);
        return;
      }
      if (req.method === 'HEAD') {
        const headRoute = matching.find((r) => r.method === 'HEAD');
        if (headRoute) {
          reqLogger.debug('Found HEAD route handler');
        } else if (matching.some((r) => r.method === 'GET')) {
          reqLogger.debug('No HEAD handler, using GET for headers');
          // Find the GET route
          const getRoute = matching.find((r) => r.method === 'GET');
          if (getRoute) {
            // Capture headers from GET handler without sending a body
            const { status, headers } = await this.captureHeadersFromHandler(
              getRoute.handler,
              req,
              sock,
            );
            sendWithContext(req, sock, status, headers, undefined);
            logRequestCompletion(reqLogger, startTime, status);
            return;
          }
        }
      }
      const route =
        matching.find((r) => r.method === req.method) || matching.find((r) => r.method === 'ANY');
      if (!route) {
        const allowed = matching.map((r) => r.method).join(',');
        reqLogger.debug('Method not allowed', { allowed });
        sendWithContext(
          req,
          sock,
          405,
          { 'Content-Type': 'text/plain', Allow: allowed },
          'Method Not Allowed',
        );
        logRequestCompletion(reqLogger, startTime, 405);
        return;
      }
      reqLogger.debug('Executing route handler', {
        method: route.method,
        path: route.originalPath,
      });
      await route.handler(req, sock);
      logRequestCompletion(reqLogger, startTime, req.ctx?.responseStatus || 200);
    } catch (err) {
      const status = 500;
      reqLogger.error('Handler error', { error: (err as Error).message });
      const isApi = req.path.startsWith('/api/');
      sendWithContext(
        req,
        sock,
        status,
        { 'Content-Type': isApi ? 'application/json' : 'text/plain' },
        isApi ? JSON.stringify({ error: 'Internal Server Error' }) : '500 Server Error',
      );
      logRequestCompletion(reqLogger, startTime, status);
    }
  }

  /**
   * Execute middleware chain
   */
  private async runMiddlewares(
    req: IncomingRequest,
    sock: Socket,
    reqLogger: Logger,
  ): Promise<void> {
    let index = 0;
    const next = async (): Promise<void> => {
      const mw = this.middlewares[index++];
      if (mw) {
        reqLogger.debug('Executing middleware', { index: index - 1 });
        await mw(req, sock, next);
      }
    };
    await next();
  }
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
  const router = new Router();
  router.use(requestIdMiddleware); // Register the request ID middleware
  router.use(optionsHandlerMiddleware); // Register OPTIONS handler middleware
  router.use(corsMiddleware); // Register the CORS middleware
  router.registerDebugRoute(); // Register the debug route
  return router;
}

// Export a singleton router instance
export default createRouter();
