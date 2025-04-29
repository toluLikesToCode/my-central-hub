import { IncomingRequest } from '../../entities/http';
import { sendResponse } from '../../entities/sendResponse';
import { Socket } from 'net';
import { saveMetrics, metricsPayloadSchema, isPerfLogArray } from './metricsService';
import { logger } from '../../utils/logger';

export const metricsController = {
  handleMetrics: async (req: IncomingRequest, sock: Socket) => {
    try {
      if (req.method !== 'POST') {
        sendResponse(
          sock,
          405,
          {
            'Content-Type': 'text/plain',
          },
          'Method Not Allowed',
        );
        return;
      }
      const payload = JSON.parse(
        typeof req.body === 'string' ? req.body : req.body?.toString() || '{}',
      );
      const clientSessionId = req.headers['x-session-id'] || req.headers['X-Session-Id'];
      // perf-only shortcut
      if (Array.isArray(payload) && isPerfLogArray(payload)) {
        await saveMetrics(payload, undefined, clientSessionId);
        sendResponse(
          sock,
          200,
          { 'Content-Type': 'application/json' },
          JSON.stringify({ status: 'ok' }),
        );
        return;
      }
      // Ensure required fields exist for validation
      if (!payload.lastRows) payload.lastRows = [];
      if (!payload.debug) payload.debug = [];
      const result = metricsPayloadSchema.safeParse(payload);
      if (!result.success) {
        logger.info(`[metrics] Invalid payload: ${JSON.stringify(result.error.format())}`);
        sendResponse(
          sock,
          400,
          {
            'Content-Type': 'application/json',
          },
          JSON.stringify({ error: 'Invalid payload', details: result.error.format() }),
        );
        return;
      }
      // validated object only
      const p = result.data;
      await saveMetrics(p, p.sessionMetrics, clientSessionId);
      sendResponse(
        sock,
        200,
        {
          'Content-Type': 'application/json',
        },
        JSON.stringify({ status: 'ok' }),
      );
    } catch (err) {
      logger.info(`[metrics] error saving payload: ${err}`);
      sendResponse(
        sock,
        500,
        {
          'Content-Type': 'text/plain',
        },
        'Internal Server Error',
      );
    }
  },
};
