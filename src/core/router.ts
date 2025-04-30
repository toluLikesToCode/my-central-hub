// src/core/router.ts
import { Socket } from 'net';
import { IncomingRequest } from '../entities/http';
import { sendResponse } from '../entities/sendResponse';
import { logger } from '../utils/logger';

/* ───── Types ─────────────────────────────────────────────────────────── */

export type Handler = (req: IncomingRequest, sock: Socket) => Promise<void> | void;
export type Middleware = (
  req: IncomingRequest,
  sock: Socket,
  next: () => Promise<void>,
) => Promise<void> | void;

interface Route {
  method: string; // 'GET' | 'POST' | 'ANY'
  regex: RegExp; // compiled path matcher
  keys: string[]; // param names (for :id stuff)
  handler: Handler;
}

/* ───── Router implementation ─────────────────────────────────────────── */

class Router {
  private middlewares: Middleware[] = [];
  private routes: Route[] = [];

  /* ---------- Middleware ---------- */
  use(mw: Middleware) {
    this.middlewares.push(mw);
  }

  /* ---------- Route registration helpers ---------- */
  add(method: string, path: string, handler: Handler) {
    const { regex, keys } = compilePath(path);
    this.routes.push({ method: method.toUpperCase(), regex, keys, handler });
  }
  get(path: string, h: Handler) {
    this.add('GET', path, h);
  }
  post(path: string, h: Handler) {
    this.add('POST', path, h);
  }
  put(path: string, h: Handler) {
    this.add('PUT', path, h);
  }
  del(path: string, h: Handler) {
    this.add('DELETE', path, h);
  }
  any(path: string, h: Handler) {
    this.add('ANY', path, h);
  }

  /* ---------- Main entry ---------- */
  async handle(req: IncomingRequest, sock: Socket): Promise<void> {
    if (req.method === 'OPTIONS') {
      sendResponse(
        sock,
        200,
        { 'Content-Type': 'text/plain', Allow: 'GET, POST, PUT, DELETE, OPTIONS' },
        'OK',
      );
      return;
    }
    if (!req.path || typeof req.path !== 'string') {
      sendResponse(
        sock,
        400,
        {
          'Content-Type':
            req.path && req.path.startsWith('/api/') ? 'application/json' : 'text/plain',
        },
        req.path && req.path.startsWith('/api/')
          ? JSON.stringify({ error: 'Bad Request' })
          : 'Bad Request',
      );
      return;
    }
    logger.info(`router saw ${req.method} ${req.path}`);
    // 1. run middleware chain
    let i = 0;
    // Use an explicit stack to unwind after handler
    const run = async (): Promise<void> => {
      if (i < this.middlewares.length) {
        const idx = i++;
        await this.middlewares[idx](req, sock, run);
        return;
      }
      // Only after all middleware, run the handler
      // 2. route lookup
      const matching = this.routes.filter((r) => r.regex.test(req.path));
      const route =
        matching.find((r) => r.method === req.method) ?? matching.find((r) => r.method === 'ANY');
      if (!route) {
        // Distinguish 404 vs 405
        if (matching.length) {
          // Only include allowed methods that are not 'ANY'
          const allowed = matching
            .map((r) => r.method)
            .filter((m) => m !== 'ANY')
            .join(', ');
          if (req.path.startsWith('/api/')) {
            sendResponse(
              sock,
              405,
              {
                'Content-Type': 'application/json',
                Allow: allowed,
              },
              JSON.stringify({ error: 'Method Not Allowed' }),
            );
          } else {
            sendResponse(sock, 405, { Allow: allowed }, 'Method Not Allowed');
          }
        } else {
          if (req.path.startsWith('/api/')) {
            sendResponse(
              sock,
              404,
              { 'Content-Type': 'application/json' },
              JSON.stringify({ error: 'Not Found' }),
            );
          } else {
            sendResponse(sock, 404, { 'Content-Type': 'text/plain' }, 'Not Found');
          }
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
      // 4. invoke handler
      try {
        await route.handler(req, sock);
      } catch (err) {
        logger.error(`Handler error: ${(err as Error).message}`);
        if (req.path.startsWith('/api/')) {
          sendResponse(
            sock,
            500,
            { 'Content-Type': 'application/json' },
            JSON.stringify({ error: 'Internal Server Error' }),
          );
        } else {
          sendResponse(sock, 500, { 'Content-Type': 'text/plain' }, '500 Server Error');
        }
      }
    };
    try {
      await run();
    } catch (err) {
      // Global error handler for middleware
      if (req.path.startsWith('/api/')) {
        sendResponse(
          sock,
          500,
          { 'Content-Type': 'application/json' },
          JSON.stringify({ error: 'Internal Server Error' }),
        );
      } else {
        sendResponse(sock, 500, { 'Content-Type': 'text/plain' }, '500 Server Error');
      }
      logger.error(`Middleware error: ${(err as Error).message}`);
      return;
    }
  }
}

/* ───── Path pattern compiler ───────────────────────────────────────────
   /files/:name  ->  ^/files/([^/]+)$           keys=['name']
   /api/*        ->  ^/api/(.*)$                keys=['*']
------------------------------------------------------------------------ */
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

export function createRouter(): Router {
  return new Router();
}

export default createRouter();
