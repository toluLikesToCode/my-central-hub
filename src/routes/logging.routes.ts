/**
 * Remote Logging Routes Module
 *
 * Provides API endpoints for log ingestion from remote clients.
 * Implements the HttpProtocolHandler logging spec from RemoteLogging.md.
 *
 * @module routes/logging.routes
 */
import router from '../core/router';
import { ingestLogs } from '../modules/logging';
import { config } from '../config/server.config';
import { Logger, FileTransport, ConsoleTransport, PrettyFormatter } from '../utils/logger';
import { getCurrentFormattedDate } from '../utils/dateFormatter';
import path from 'path';
import os from 'os';

// Create a specialized logger for logging routes
const loggingRoutesLogger = new Logger({
  metadata: {
    module: 'routes',
    component: 'logging-routes',
    server: os.hostname(),
    version: '1.0.0',
  },
  transports: [
    new ConsoleTransport({
      formatter: new PrettyFormatter({
        useColors: true,
        useBoxes: false,
        showTimestamp: true,
      }),
      level: 'info',
    }),
    new FileTransport({
      filename: path.join(config.logging.logDir, 'logging-routes.log'),
      formatter: new PrettyFormatter({
        useColors: false,
        showTimestamp: true,
      }),
      level: 'debug',
    }),
  ],
});

// Log the initialization of logging routes
loggingRoutesLogger.info('Initializing remote logging routes', {
  timestamp: getCurrentFormattedDate(),
});

// Register the log ingestion route
router.post('/api/logs/ingest', async (req, sock) => {
  loggingRoutesLogger.debug('Received log ingestion request', {
    ip: sock.remoteAddress,
    contentLength: req.headers['content-length'],
  });

  await ingestLogs(req, sock);
});

// Log completion of route registration
loggingRoutesLogger.success('Remote logging routes registered successfully', {
  endpoints: ['/api/logs/ingest'],
  methods: ['POST'],
  timestamp: getCurrentFormattedDate(),
});

export {};
