import { IncomingRequest } from '../../../entities/http';
import { sendResponse } from '../../../entities/sendResponse';
import { Socket } from 'net';
import { saveMetrics, isPerfLogArray } from './metricsService';
import logger from '../../../utils/logger';

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
      let payload: unknown;
      try {
        // Accept Buffer, string, or already-parsed object
        if (Buffer.isBuffer(req.body)) {
          payload = JSON.parse(req.body.toString('utf8'));
        } else if (typeof req.body === 'string') {
          payload = JSON.parse(req.body);
        } else if (typeof req.body === 'object' && req.body !== null) {
          payload = req.body;
        } else {
          payload = {};
        }
      } catch (err) {
        logger.info(`[metrics] Invalid JSON: ${err}`);
        sendResponse(
          sock,
          400,
          { 'Content-Type': 'application/json' },
          JSON.stringify({ error: 'Invalid JSON', details: String(err) }),
        );
        return;
      }
      const clientSessionId = req.headers['x-session-id'] || req.headers['X-Session-Id'];
      // PerfLog batch shortcut
      if (Array.isArray(payload) && isPerfLogArray(payload)) {
        await saveMetrics(payload, clientSessionId);
        sendResponse(
          sock,
          200,
          { 'Content-Type': 'application/json' },
          JSON.stringify({ status: 'OK', payload: payload, message: 'valid payload' }),
        );
        return;
      }
      // Always call saveMetrics, even if partially invalid
      await saveMetrics(payload, clientSessionId);
      sendResponse(
        sock,
        200,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ status: 'OK', payload: payload, message: 'possible invalid payload' }),
      );
    } catch (err) {
      logger.info(`[metrics] error saving payload: ${err}`);
      sendResponse(
        sock,
        500,
        {
          'Content-Type': 'text/plain',
        },
        Buffer.from(JSON.stringify({ message: 'Internal Server Error', error: err })),
      );
    }
  },
};
