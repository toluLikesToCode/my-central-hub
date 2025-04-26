import { Socket } from 'net';
import { IncomingRequest } from '../entities/http';
import { fileStreamingController } from '../modules/file-streamer/index';
import { logger } from '../utils/logger';
import { URL } from 'url';

type Handler = (req: IncomingRequest, socket: Socket) => Promise<void>;

interface Route {
  method: string;
  path: string;
  handler: Handler;
}

class Router {
  private routes: Route[] = [];

  constructor() {
    this.initializeRoutes();
  }

  private initializeRoutes() {
    this.routes.push({
      method: 'GET',
      path: '/stream',
      handler: fileStreamingController.handleStream,
    });

    this.routes.push({
      method: 'GET',
      path: '/files',
      handler: fileStreamingController.listFiles,
    });
  }

  async handle(req: IncomingRequest, socket: Socket): Promise<void> {
    logger.info(`Routing request: ${req.method} ${req.path}`);

    if (!req.path) {
      logger.warn(`No path provided in request`);
      socket.write('HTTP/1.1 400 Bad Request\r\n\r\nMissing path');
      socket.end();
      return;
    }

    const url = new URL(req.path, `http://${req.headers.host}`);
    const pathname = url.pathname;

    const route = this.routes.find((r) => r.method === req.method && r.path === pathname);

    if (!route) {
      logger.warn(`No route found for ${req.method} ${pathname}`);
      socket.write('HTTP/1.1 404 Not Found\r\n\r\n404 Not Found');
      socket.end();
      return;
    }

    try {
      await route.handler(req, socket);
    } catch (error) {
      logger.error(`Handler error: ${(error as Error).message}`);
      socket.write('HTTP/1.1 500 Internal Server Error\r\n\r\nServer Error');
      socket.end();
    }
  }
}

export const router = new Router();
