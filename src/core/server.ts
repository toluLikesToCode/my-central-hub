import { createServer, Socket } from 'net';
import { HttpRequestParser } from './httpParser';
import { router } from './router';
import { logger } from '../utils/logger';
import { sendResponse } from '../entities/sendResponse';
import { config } from '../config/server.config'; // Assuming config is imported from a config file

export class HttpServer {
  private server = createServer();
  private readonly connections = new Set<Socket>();

  constructor(private port: number) {
    this.setupServer();
  }

  private setupServer() {
    this.server.on('connection', (socket: Socket) => {
      this.connections.add(socket);
      const parser = new HttpRequestParser();

      // --- ⏰ Idle Timeout (Protection) ---
      const HEADER_TIMEOUT_MS = config.headerTimeoutMs;
      const BODY_TIMEOUT_MS = config.bodyTimeoutMs;
      let headerTimer: NodeJS.Timeout | undefined;
      let bodyTimer: NodeJS.Timeout | undefined;

      const refreshTimeout = () => {
        if (headerTimer) clearTimeout(headerTimer);
        headerTimer = setTimeout(() => {
          logger.warn('Closing idle socket (header timeout)');
          socket.destroy();
        }, HEADER_TIMEOUT_MS);
      };

      const refreshBodyTimeout = () => {
        if (bodyTimer) clearTimeout(bodyTimer);
        bodyTimer = setTimeout(() => {
          logger.warn('Closing idle socket (body timeout)');
          socket.destroy();
        }, BODY_TIMEOUT_MS);
      };

      refreshTimeout(); // start immediately

      socket.once('close', () => {
        this.connections.delete(socket);
        if (headerTimer) clearTimeout(headerTimer);
        if (bodyTimer) clearTimeout(bodyTimer);
      });
      logger.info('New connection established.');
      socket.on('data', async (chunk: Buffer) => {
        refreshTimeout();
        try {
          // First feed yields the first complete request (or null)
          let req = parser.feed(chunk);
          if (!req) return; // need more bytes

          clearTimeout(headerTimer);

          // Handle all pipelined requests in buffer
          do {
            // If this is a body-bearing method, start body timeout
            if (req.method === 'POST' || req.method === 'PUT' || req.method === 'PATCH') {
              refreshBodyTimeout();
            }

            await router.handle(req, socket);
            clearTimeout(bodyTimer);

            // now pull the next request from any leftover bytes
            req = parser.feed(Buffer.alloc(0));
          } while (req);

          const pending = parser.getPendingBytes();
          if (pending > 0) {
            // more pipelined data waiting—keep alive
            refreshTimeout();
          } else {
            socket.end();
          }
        } catch (err) {
          logger.error(`Failed request: ${(err as Error).message}`);
          sendResponse(socket, 400, { 'Content-Type': 'text/plain' }, 'Bad Request');
        }
      });

      socket.on('error', (err) => {
        logger.error(`Socket error: ${err.message}`);
      });
    });

    this.server.on('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'EADDRINUSE') {
        this.port += 1; // try the next port
        logger.warn(`Port busy, retrying on ${this.port}`);
        this.server.listen(this.port);
      } else {
        logger.error(`Server error: ${err.message}`);
      }
    });
  }

  /**
   * Gracefully shuts down the server and every open TCP socket.
   */
  public async stop(): Promise<void> {
    logger.info('🛑  Shutting down HTTP server');
    for (const sock of this.connections) sock.destroy();
    await new Promise<void>((resolve, reject) =>
      this.server.close((err) => (err ? reject(err) : resolve())),
    );
  }

  /**
   * Public method to destroy all active sockets.
   */
  public destroySockets(): void {
    this.connections.forEach((socket) => socket.destroy());
  }

  public start() {
    this.server.listen(this.port, () => {
      logger.info(`🚀 Server running at port ${this.port}`);
    });

    // graceful shutdown on Ctrl-C / kill
    ['SIGINT', 'SIGTERM'].forEach((sig) =>
      process.on(sig as NodeJS.Signals, () => {
        this.stop()
          .then(() => process.exit(0))
          .catch(() => process.exit(1));
      }),
    );

    // src/core/server.ts  – inside start() after existing SIGINT/SIGTERM hooks
    ['SIGUSR2'].forEach((sig) =>
      process.once(sig as NodeJS.Signals, () => {
        this.stop().then(() => process.kill(process.pid, sig));
      }),
    );
  }
}
