/* eslint-disable @typescript-eslint/no-unused-vars */
/**
 * Main Entry Point for My Central Hub
 *
 * This module initializes the application server and loads all routes.
 * It sets up logging, error handling, and starts the HTTP server with
 * the configured settings from server.config.ts.
 *
 * @module main
 */
import { HttpServer } from './core/server';
import { config } from './config/server.config';
import { Logger, ConsoleTransport, PrettyFormatter, FileTransport } from './utils/logger';
import path from 'path';
import process from 'process';

// Create a more comprehensive logger with both console and file transports
const logger = new Logger({
  transports: [
    new ConsoleTransport({
      formatter: new PrettyFormatter({
        useBoxes: false,
        useColors: true,
        showTimestamp: false,
        indent: 3,
        arrayLengthLimit: 15,
        objectKeysLimit: 10,
        maxDepth: 4,
        stringLengthLimit: 300,
      }),
    }),
    // Add a file transport specifically for application startup logs
    new FileTransport({
      filename: path.join(config.logging.logDir, 'startup.log'),
      formatter: new PrettyFormatter({
        useColors: false,
        useBoxes: false,
        showTimestamp: true,
      }),
    }),
  ],
});

/**
 * Application startup sequence
 */
async function startApplication() {
  try {
    // Log detailed startup information
    logger.info('Starting My Central Hub server', {
      environment: process.env.NODE_ENV || 'development',
      timestamp: new Date().toISOString(),
      nodeVersion: process.version,
    });

    // Log configuration details
    logger.info('Server configuration loaded', config);

    // Log enabled features
    logger.info('Enabled features', {
      features: Object.entries(config.features)
        .filter(([_, enabled]) => enabled)
        .map(([feature]) => feature),
    });

    // Create and start HTTP server
    const server = new HttpServer(config.port);

    // Log routes status
    logger.info('Routes registered and loaded');

    // Start the server
    await server.start();

    // Log successful startup
    logger.success('Server started successfully', {
      port: config.port,
      publicDir: config.publicDir,
    });

    // Set up graceful shutdown
    setupGracefulShutdown(server);
  } catch (error) {
    logger.error('Failed to start server', {
      error:
        error instanceof Error
          ? {
              message: error.message,
              stack: error.stack,
            }
          : String(error),
    });
    process.exit(1);
  }
}

/**
 * Set up handlers for graceful shutdown
 * @param server - The HTTP server instance
 */
function setupGracefulShutdown(server: HttpServer) {
  const shutdown = async () => {
    logger.info('Shutting down server gracefully...');
    try {
      await server.stop();
      logger.info('Server stopped successfully');
      await logger.close();
      process.exit(0);
    } catch (error) {
      logger.error('Error during server shutdown', {
        error: error instanceof Error ? error.message : String(error),
      });
      process.exit(1);
    }
  };

  // Handle termination signals
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);

  // Handle uncaught exceptions
  process.on('uncaughtException', (error) => {
    logger.error('Uncaught exception', {
      error: {
        message: error.message,
        stack: error.stack,
      },
    });
    shutdown().catch(() => process.exit(1));
  });

  // Handle unhandled promise rejections
  process.on('unhandledRejection', (reason) => {
    logger.error('Unhandled promise rejection', {
      reason:
        reason instanceof Error
          ? {
              message: reason.message,
              stack: reason.stack,
            }
          : String(reason),
    });
  });
}

// Start the application
startApplication();
