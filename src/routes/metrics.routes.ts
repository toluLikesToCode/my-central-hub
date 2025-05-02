// routes/metrics.routes.ts
import router from '../core/router';
import { metricsController } from '../modules/app-metrics/app_gallery-generator';
import { config } from '../config/server.config';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { sendResponse as sendCustomResponse } from '../entities/sendResponse';
import { Readable } from 'stream';
import { Socket } from 'net';

/**
 * Accepts the JSON payload, validates, and writes into SQLite.
 */

// CORS middleware for all /api/metrics/* routes
router.use(async (req, sock, next) => {
  if (req.path.startsWith('/api/metrics/')) {
    const headers = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, X-Session-Id',
    };
    if (req.method === 'OPTIONS') {
      // Preflight request
      sock.write(
        [
          'HTTP/1.1 204 No Content',
          ...Object.entries(headers).map(([k, v]) => `${k}: ${v}`),
          '',
          '',
        ].join('\r\n'),
      );
      sock.end();
      return;
    } else {
      // For normal requests, add CORS headers to response
      req.ctx = req.ctx || {};
      req.ctx.corsHeaders = headers;
    }
  }
  await next();
});

if (config.features.metrics) {
  router.post('/api/metrics/gallery-generator', async (req, sock) => {
    // Define the custom response sender *locally*
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const sendCustomResponse = (
      sock: Socket,
      status: number,
      headers: Record<string, string>,
      body?: string | Buffer | Readable,
    ) => {
      // Access CORS headers from the request context (set by middleware)
      const corsHeaders = req.ctx && req.ctx.corsHeaders ? req.ctx.corsHeaders : {};
      sendCustomResponse(sock, status, { ...headers, ...corsHeaders }, body);
    };
    // Patch sendResponse for this request
    //req.sendResponse = sendResponse;
    await metricsController.handleMetrics(req, sock);
  });
}
