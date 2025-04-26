import { Socket } from 'net';
import { IncomingRequest } from '../entities/http';
import { fileStreamingController } from '../modules/file-streamer/index';
import { logger } from '../utils/logger';
import { URL } from 'url';
import { sendResponse } from '../entities/sendResponse';

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
      sendResponse(socket, 400, { 'Content-Type': 'text/plain' }, 'Missing path');
      return;
    }

    const url = new URL(req.path, `http://${req.headers.host}`);
    const pathname = url.pathname;

    const route = this.routes.find((r) => r.method === req.method && r.path === pathname);

    if (!route) {
      logger.warn(`No route found for ${req.method} ${pathname}`);
      sendResponse(socket, 404, { 'Content-Type': 'text/plain' }, '404 Not Found');
      return;
    }

    try {
      await route.handler(req, socket);
    } catch (error) {
      logger.error(`Handler error: ${(error as Error).message}`);
      sendResponse(socket, 500, { 'Content-Type': 'text/plain' }, 'Server Error');
    }
  }
}

export const router = new Router();
