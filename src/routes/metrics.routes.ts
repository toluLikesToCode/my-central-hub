/**
 * Metrics Routes Module
 *
 * Provides API endpoints for collecting and accessing metrics from various applications
 * in the Central Hub ecosystem. Each app has its own metrics endpoint.
 *
 * @module routes/metrics.routes
 */
import router from '../core/router';
import { appGalleryGenerator } from '../modules/app-metrics';
import { config } from '../config/server.config';
import { Logger, FileTransport, ConsoleTransport, PrettyFormatter } from '../utils/logger';
import { formatDate } from '../utils/dateFormatter';
import path from 'path';
import os from 'os';
import { beginChunkedResponse, sendResponse } from '../entities/sendResponse';

// Create a specialized logger for metrics routes with request tracking
const metricsRouteLogger = new Logger({
  metadata: {
    module: 'routes',
    component: 'metrics-routes',
    server: os.hostname(),
    version: '2.0.0',
  },
  transports: [
    new ConsoleTransport({
      formatter: new PrettyFormatter({
        useColors: true,
        useBoxes: true,
        showTimestamp: true,
      }),
      level: 'info',
    }),
    new FileTransport({
      filename: path.join(config.logging.logDir, 'metrics-routes.log'),
      formatter: new PrettyFormatter({
        useColors: false,
        showTimestamp: true,
      }),
      level: 'debug',
    }),
  ],
});

// System startup timestamp using human-readable format
const serverStartTime = formatDate(new Date());

// Log the initialization of metrics routes
metricsRouteLogger.info('Initializing metrics routes', {
  enabled: config.features.metrics,
  startupTime: serverStartTime,
  environment: process.env.NODE_ENV || 'development',
  metricsConfig: {
    dbPath: config.dbPath,
    logDir: config.logging.logDir,
  },
});

if (config.features.metrics) {
  /**
   * Gallery Generator App Metrics (POST)
   *
   * Accepts metrics data from the Gallery Generator application.
   * This includes: performance logs, user engagement data, debug logs, and session metrics.
   * Data is validated and stored in the SQLite database.
   */
  router.post('/api/metrics/gallery-generator', (req, sock) => {
    const startTime = Date.now();
    const clientIp = req.headers['x-forwarded-for'] || sock.remoteAddress || 'unknown';
    const requestId =
      req.ctx?.requestId?.toString() ||
      `req_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
    metricsRouteLogger.debug('Gallery metrics request received', {
      requestId,
      remoteAddress: clientIp,
      contentLength: req.headers['content-length'],
      timestamp: formatDate(new Date()),
      method: req.method,
    });

    appGalleryGenerator.metricsController
      .handleMetrics(req, sock)
      .then(() => {
        const duration = Date.now() - startTime;

        // Only log successful requests at info level if they take longer than expected
        if (duration > 500) {
          metricsRouteLogger.info('Gallery metrics request processed (slow)', {
            requestId,
            processingTime: `${duration}ms`,
            client: clientIp,
            sessionId: req.headers['x-session-id'] || 'unknown',
          });
        } else {
          metricsRouteLogger.debug('Gallery metrics request processed', {
            requestId,
            processingTime: `${duration}ms`,
            client: clientIp,
          });
        }
      })
      .catch((err) => {
        const duration = Date.now() - startTime;
        metricsRouteLogger.error('Error processing gallery metrics request', {
          requestId,
          error: err.message,
          stack: err.stack,
          processingTime: `${duration}ms`,
          client: clientIp,
        });
      });
  });

  /**
   * Gallery Generator Recent Sessions (GET)
   *
   * Retrieves a list of recent metrics sessions for analysis and debugging.
   * Returns session IDs, timestamps, and counts of various metrics.
   */
  router.get('/api/metrics/gallery-generator/sessions', async (req, sock) => {
    const startTime = Date.now();
    const clientIp = req.headers['x-forwarded-for'] || sock.remoteAddress || 'unknown';
    const requestId =
      req.ctx?.requestId?.toString() ||
      `req_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;

    metricsRouteLogger.debug('Session list request received', {
      requestId,
      remoteAddress: clientIp,
      timestamp: formatDate(new Date()),
      method: req.method,
    });

    try {
      // Set socket timeout to a much higher value for this long-running operation
      if (typeof sock.setTimeout === 'function') {
        sock.setTimeout(300000); // 5 minute timeout
      }

      // Begin a chunked response to prevent socket hang up
      const responseHandler = beginChunkedResponse(sock, {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
        },
        requestId,
        noBuffering: true, // Prevent proxy buffering
      });

      // Send a progress notification to keep connection alive before the main query starts
      responseHandler.sendChunk({
        status: 'processing',
        message: 'Fetching session data...',
        timestamp: formatDate(new Date()),
      });

      // Default limit to 20, but allow query param override with bounds checking
      const limitParam = req.query?.limit;
      let limit = 20; // Default
      if (limitParam) {
        const parsedLimit = parseInt(limitParam as string, 10);
        // Enforce reasonable bounds for limit (1-100)
        limit = isNaN(parsedLimit) ? 20 : Math.min(Math.max(parsedLimit, 1), 100);
      }

      // Get recent sessions with the optimized implementation
      const sessions = await appGalleryGenerator.getRecentSessions(limit);

      const duration = Date.now() - startTime;
      metricsRouteLogger.info('Sessions list request processed', {
        requestId,
        sessionCount: sessions.length,
        processingTime: `${duration}ms`,
        client: clientIp,
      });

      // Send the actual response data
      responseHandler.sendChunk({
        sessions,
        count: sessions.length,
        timestamp: formatDate(new Date()),
        serverTime: formatDate(new Date()),
        processingTime: `${duration}ms`,
      });

      // End the chunked response
      responseHandler.endResponse();
    } catch (err) {
      const error = err as Error;
      const duration = Date.now() - startTime;

      metricsRouteLogger.error('Error retrieving sessions list', {
        requestId,
        error: error.message,
        stack: error.stack,
        processingTime: `${duration}ms`,
        client: clientIp,
      });

      // If we haven't written headers yet, send an error response
      if (!sock.writableEnded) {
        // Send error response with standard headers
        sendResponse(
          sock,
          500,
          {
            'Content-Type': 'application/json',
            'X-Request-ID': requestId,
          },
          JSON.stringify({
            error: 'Failed to retrieve sessions',
            message: error.message,
            timestamp: formatDate(new Date()),
          }),
        );
      }
    }
  });

  /**
   * Gallery Generator Session Performance Stats (GET)
   *
   * Retrieves detailed performance statistics for a specific session.
   * Includes min/max/avg timing metrics for various actions.
   */
  router.get(
    '/api/metrics/gallery-generator/sessions/:sessionId/performance',
    async (req, sock) => {
      const startTime = Date.now();
      const clientIp = req.headers['x-forwarded-for'] || sock.remoteAddress || 'unknown';
      const requestId =
        req.ctx?.requestId?.toString() ||
        `req_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;

      // Extract sessionId from URL path
      const pathMatch = req.path.match(
        /\/api\/metrics\/gallery-generator\/sessions\/([^/]+)\/performance$/,
      );
      const sessionId = pathMatch ? pathMatch[1] : '';

      metricsRouteLogger.debug('Session performance stats request received', {
        requestId,
        sessionId,
        remoteAddress: clientIp,
        timestamp: formatDate(new Date()),
        method: req.method,
      });

      if (!sessionId) {
        metricsRouteLogger.warn('Missing sessionId in performance stats request', {
          requestId,
          path: req.path,
          client: clientIp,
        });

        // Use sendResponse utility for consistent error handling
        sendResponse(
          sock,
          400,
          {
            'Content-Type': 'application/json',
            'X-Request-ID': requestId,
          },
          JSON.stringify({
            error: 'Missing session ID',
            message: 'Session ID is required in the URL path',
            timestamp: formatDate(new Date()),
          }),
        );
        return;
      }

      try {
        // Extend socket timeout for this long-running database operation
        if (typeof sock.setTimeout === 'function') {
          sock.setTimeout(300000); // 5 minute timeout for consistency with the sessions endpoint
        }

        // For large sessions, use chunked response to avoid timeouts
        const hasLargeStats = sessionId.startsWith('large_') || req.query.enhanced === 'true';

        if (hasLargeStats) {
          // Begin chunked response for potentially large responses
          const responseHandler = beginChunkedResponse(sock, {
            status: 200,
            headers: {
              'Content-Type': 'application/json',
              'X-Processing-Mode': 'chunked',
            },
            requestId,
            noBuffering: true,
          });

          // Send initial progress message
          responseHandler.sendChunk({
            status: 'processing',
            message: 'Fetching performance stats...',
            sessionId,
            timestamp: formatDate(new Date()),
          });

          // Get performance stats for the session
          const stats = await appGalleryGenerator.getSessionPerformanceStats(sessionId);

          const duration = Date.now() - startTime;
          metricsRouteLogger.info('Session performance stats request processed (chunked)', {
            requestId,
            sessionId,
            actionCount: stats.distinctActions,
            processingTime: `${duration}ms`,
            responseMode: 'chunked',
            client: clientIp,
          });

          // Send the full stats response
          responseHandler.sendChunk({
            ...stats,
            timestamp: formatDate(new Date()),
            serverTime: formatDate(new Date()),
            processingTime: `${duration}ms`,
          });

          // End the chunked response
          responseHandler.endResponse();
        } else {
          // Standard response for normal-sized sessions
          // Get performance stats for the session
          const stats = await appGalleryGenerator.getSessionPerformanceStats(sessionId);

          const duration = Date.now() - startTime;
          metricsRouteLogger.info('Session performance stats request processed', {
            requestId,
            sessionId,
            actionCount: stats.distinctActions,
            processingTime: `${duration}ms`,
            client: clientIp,
          });

          // Use sendResponse utility for consistent response handling
          sendResponse(
            sock,
            200,
            {
              'Content-Type': 'application/json',
              'X-Request-ID': requestId,
              'X-Processing-Time': `${duration}ms`,
            },
            JSON.stringify({
              ...stats,
              timestamp: formatDate(new Date()),
              serverTime: formatDate(new Date()),
              processingTime: `${duration}ms`,
            }),
          );
        }
      } catch (err) {
        const error = err as Error;
        const duration = Date.now() - startTime;

        metricsRouteLogger.error('Error retrieving session performance stats', {
          requestId,
          sessionId,
          error: error.message,
          stack: error.stack,
          processingTime: `${duration}ms`,
          client: clientIp,
        });

        // Use sendResponse utility for consistent error handling
        if (!sock.writableEnded && !sock.destroyed) {
          sendResponse(
            sock,
            500,
            {
              'Content-Type': 'application/json',
              'X-Request-ID': requestId,
            },
            JSON.stringify({
              error: 'Failed to retrieve session performance stats',
              message: error.message,
              timestamp: formatDate(new Date()),
            }),
          );
        } else {
          // Headers already sent via chunked response - send error as a chunk
          try {
            const responseHandler = {
              sendChunk: (data: unknown) => {
                if (!sock.destroyed) {
                  const chunk = JSON.stringify(data);
                  const chunkLength = Buffer.byteLength(chunk).toString(16);
                  sock.write(`${chunkLength}\r\n${chunk}\r\n`);
                }
              },
              endResponse: () => {
                if (!sock.destroyed) {
                  sock.write('0\r\n\r\n');
                  sock.end();
                }
              },
            };

            responseHandler.sendChunk({
              error: 'Failed to retrieve session performance stats',
              message: error.message,
              timestamp: formatDate(new Date()),
            });

            responseHandler.endResponse();
          } catch {
            if (!sock.destroyed) {
              sock.end();
            }
          }
        }
      }
    },
  );

  // Log successful routes setup
  metricsRouteLogger.success('Metrics routes registered successfully', {
    routes: [
      { method: 'POST', path: '/api/metrics/gallery-generator' },
      { method: 'GET', path: '/api/metrics/gallery-generator/sessions' },
      { method: 'GET', path: '/api/metrics/gallery-generator/sessions/:sessionId/performance' },
    ],
    registrationTime: formatDate(new Date()),
  });
} else {
  metricsRouteLogger.warn('Metrics feature is disabled in configuration', {
    config: 'config.features.metrics',
    enableInstructions: 'Update server.config.ts to enable this feature',
  });
}
