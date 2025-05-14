/* eslint-disable @typescript-eslint/no-explicit-any */
import { IncomingRequest } from '../../../entities/http';
import { sendWithContext } from '../../../entities/sendResponse';
import { Socket } from 'net';
import { saveMetrics, isPerfLogArray } from './metricsService';
import { Logger, FileTransport, ConsoleTransport, PrettyFormatter } from '../../../utils/logger';
import { formatDate } from '../../../utils/dateFormatter';
import path from 'path';
import { config } from '../../../config/server.config';

// Create dedicated logger for metrics controller with enhanced formatting
const metricsLogger = new Logger({
  metadata: {
    module: 'app-metrics',
    component: 'gallery-generator-controller',
    service: 'metrics',
  },
  transports: [
    new ConsoleTransport({
      formatter: new PrettyFormatter({
        useColors: true,
        useBoxes: false,
        showTimestamp: true,
        maxDepth: 3,
      }),
      level: 'info',
    }),
    new FileTransport({
      filename: path.join(config.logging.logDir, 'metrics-controller.log'),
      formatter: new PrettyFormatter({
        useColors: false,
        useBoxes: false,
        showTimestamp: true,
        indent: 2,
        maxDepth: 3,
        arrayLengthLimit: 10,
        objectKeysLimit: 8,
        stringLengthLimit: 200,
      }),
      level: 'debug',
    }),
  ],
});

export const metricsController = {
  /**
   * Handles incoming metrics data from the Gallery Generator app.
   * Supports both performance log batches and full payload metrics.
   *
   * @param req - The incoming HTTP request containing metrics data
   * @param sock - The TCP socket to write the response to
   * @returns {Promise<void>} Resolves when metrics processing is complete
   */
  handleMetrics: async (req: IncomingRequest, sock: Socket): Promise<void> => {
    const requestStart = Date.now();
    const clientIp = req.headers['x-forwarded-for'] || sock.remoteAddress || 'unknown';
    const clientSessionId = req.headers['x-session-id'] || req.headers['X-Session-Id'];
    const contentLength = req.headers['content-length']
      ? parseInt(req.headers['content-length'], 10)
      : req.body
        ? Buffer.byteLength(JSON.stringify(req.body))
        : 0;

    // Log the incoming metrics request with detailed metadata
    metricsLogger.info(`Gallery metrics request received`, {
      requestTime: formatDate(new Date()),
      clientIp,
      clientSessionId,
      contentLength: `${(contentLength / 1024).toFixed(2)} KB`,
      contentType: req.headers['content-type'],
      userAgent: req.headers['user-agent'],
    });

    try {
      if (req.method !== 'POST') {
        metricsLogger.warn(`Invalid method ${req.method} for metrics endpoint`, {
          method: req.method,
          path: req.path,
          clientIp,
        });

        sendWithContext(
          req,
          sock,
          405,
          {
            'Content-Type': 'text/plain',
          },
          'Method Not Allowed',
        );
        return;
      }

      let payload: unknown;
      let payloadType: string = 'unknown';

      try {
        // Accept Buffer, string, or already-parsed object
        if (Buffer.isBuffer(req.body)) {
          payload = JSON.parse(req.body.toString('utf8'));
          payloadType = 'buffer';
        } else if (typeof req.body === 'string') {
          payload = JSON.parse(req.body);
          payloadType = 'string';
        } else if (typeof req.body === 'object' && req.body !== null) {
          payload = req.body;
          payloadType = 'object';
        } else {
          payload = {};
          payloadType = 'empty';
          metricsLogger.warn(`Empty or invalid payload received`, {
            bodyType: typeof req.body,
            clientSessionId,
          });
        }
      } catch (err) {
        const parseError = err as Error;
        metricsLogger.error(`Failed to parse metrics JSON payload`, {
          error: parseError.message,
          stack: parseError.stack,
          bodyPreview: Buffer.isBuffer(req.body)
            ? req.body.toString('utf8').substring(0, 200) + '...'
            : String(req.body).substring(0, 200) + '...',
          clientSessionId,
          clientIp,
        });

        sendWithContext(
          req,
          sock,
          400,
          { 'Content-Type': 'application/json' },
          JSON.stringify({
            error: 'Invalid JSON',
            details: parseError.message,
            timestamp: formatDate(new Date()),
          }),
        );
        return;
      }

      // Detailed payload logging based on type
      if (Array.isArray(payload)) {
        if (isPerfLogArray(payload)) {
          // Performance log batch detected
          const metricActions = payload
            .map((entry) => entry.action)
            .reduce(
              (acc, curr) => {
                acc[curr] = (acc[curr] || 0) + 1;
                return acc;
              },
              {} as Record<string, number>,
            );

          metricsLogger.info(`Processing performance log batch`, {
            batchSize: payload.length,
            sessionId: clientSessionId,
            actionCounts: metricActions,
            firstTimestamp: payload[0]?.timestamp,
            lastTimestamp: payload[payload.length - 1]?.timestamp,
          });

          await saveMetrics(payload, clientSessionId);

          const duration = Date.now() - requestStart;
          metricsLogger.success(`Performance log batch processed successfully`, {
            batchSize: payload.length,
            processingTime: `${duration}ms`,
            throughput: `${Math.round(payload.length / (duration / 1000))} items/sec`,
            sessionId: clientSessionId,
          });

          sendWithContext(
            req,
            sock,
            200,
            { 'Content-Type': 'application/json' },
            JSON.stringify({
              status: 'OK',
              message: 'Performance log batch processed',
              itemCount: payload.length,
              processingTime: `${duration}ms`,
            }),
          );
          return;
        } else {
          metricsLogger.warn(`Received array payload that is not a valid PerfLog array`, {
            arrayLength: payload.length,
            sessionId: clientSessionId,
            sampleItem: payload[0],
          });
        }
      } else if (typeof payload === 'object' && payload !== null) {
        // Full metrics payload detected - summarize important data points
        const payloadObj = payload as Record<string, any>;

        const summary = {
          sessionId: clientSessionId || payloadObj.sessionMetrics?.sessionId,
          timestamp: payloadObj.timestamp,
          sessionStart: payloadObj.sessionStart
            ? formatDate(new Date(payloadObj.sessionStart))
            : undefined,
          engagementItemCount: payloadObj.engagement
            ? Object.keys(payloadObj.engagement).length
            : 0,
          perfLogCount: Array.isArray(payloadObj.perfLog) ? payloadObj.perfLog.length : 0,
          debugLogCount: Array.isArray(payloadObj.debug) ? payloadObj.debug.length : 0,
          hasSessionMetrics: !!payloadObj.sessionMetrics,
          userMetrics: payloadObj.sessionMetrics
            ? {
                scrollDistance: payloadObj.sessionMetrics.scrollDistance,
                itemsLoaded: payloadObj.sessionMetrics.itemsLoaded,
                infiniteScrollLoads: payloadObj.sessionMetrics.infiniteScrollLoads,
                modalsOpened: payloadObj.sessionMetrics.modalsOpened,
                sessionDuration: payloadObj.sessionMetrics.startTime
                  ? `${Math.round((Date.now() - payloadObj.sessionMetrics.startTime) / 1000 / 60)} minutes`
                  : 'unknown',
              }
            : undefined,
          videoMetrics: payloadObj.sessionMetrics?.videoMetrics
            ? {
                plays: payloadObj.sessionMetrics.videoMetrics.plays,
                completions: payloadObj.sessionMetrics.videoMetrics.completions,
                watchTime: `${Math.round(payloadObj.sessionMetrics.videoMetrics.watchTime / 1000)} seconds`,
              }
            : undefined,
        };

        metricsLogger.info(`Processing full metrics payload`, summary);
      }

      // Process all payload types
      await saveMetrics(payload, clientSessionId);

      const duration = Date.now() - requestStart;
      metricsLogger.success(`Metrics payload processed successfully`, {
        payloadType,
        processingTime: `${duration}ms`,
        sessionId: clientSessionId,
        contentSize: `${(contentLength / 1024).toFixed(2)} KB`,
      });

      sendWithContext(
        req,
        sock,
        200,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          status: 'OK',
          message: 'Metrics payload processed',
          processingTime: `${duration}ms`,
          timestamp: formatDate(new Date()),
        }),
      );
    } catch (err) {
      const error = err as Error;
      const duration = Date.now() - requestStart;

      metricsLogger.error(`Failed to process metrics payload`, {
        error: error.message,
        stack: error.stack,
        clientSessionId,
        clientIp,
        processingTime: `${duration}ms`,
      });

      sendWithContext(
        req,
        sock,
        500,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          message: 'Internal Server Error',
          error: error.message,
          timestamp: formatDate(new Date()),
        }),
      );
    }
  },
};
