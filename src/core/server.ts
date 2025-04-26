import { createServer, Socket } from 'net';
import { parser } from './parser';
import { router } from './router';
import { logger } from '../utils/logger';
import { sendResponse } from '../entities/sendResponse';

export class HttpServer {
  private server = createServer();
  private readonly connections = new Set<Socket>();

  constructor(private readonly port: number) {
    this.setupServer();
  }

  private setupServer() {
    this.server.on('connection', (socket: Socket) => {
      this.connections.add(socket);
      socket.once('close', () => this.connections.delete(socket));
      logger.info('New connection established.');
      socket.on('data', async (chunk: Buffer) => {
        try {
          const request = parser.parse(chunk.toString());
          await router.handle(request, socket);
          // router handles response sending
        } catch (error) {
          logger.error(`Failed to process request: ${(error as Error).message}`);
          sendResponse(socket, 400, { 'Content-Type': 'text/plain' }, 'Bad Request');
        }
      });

      socket.on('error', (err) => {
        logger.error(`Socket error: ${err.message}`);
      });
    });

    this.server.on('error', (err) => {
      logger.error(`Server error: ${err.message}`);
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
  }
}
