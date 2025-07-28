/**
 * Stream Routes Module
 *
 * @deprecated This module is deprecated and all routes will be removed in the next major version.
 * Please use the file-hosting routes for all file operations:
 * - GET /api/files?file=filename - For retrieving files
 * - GET /api/files/:filename - For retrieving files (RESTful)
 * - POST /api/files - For uploading files
 * - DELETE /api/files/:filename - For deleting files
 *
 * @module routes/stream.routes
 */
import router from '../core/router';
import { config } from '../config/server.config';
import logger from '../utils/logger';
import { sendWithContext } from '../entities/sendResponse';
import { Socket } from 'net';
import { IncomingRequest } from '../entities/http';
import { formatDate } from '../utils/dateFormatter';

// Create a module-specific logger with metadata; if logger.child is unavailable (e.g. auto-mocked), use no-op
const streamLogger =
  logger && typeof logger.child === 'function'
    ? logger.child({ module: 'stream-routes', feature: 'media-streaming', deprecated: true })
    : { warn: (): void => {}, info: (): void => {}, error: (): void => {} };

// Log when stream routes are registered during startup
streamLogger.warn('Registering deprecated streaming routes', {
  deprecationNotice: 'These routes are deprecated. Use /api/files routes instead.',
  enabled: config.features.fileStreaming,
  mediaDirectory: config.mediaDir,
  replacementRoutes: ['/api/files?file=filename', '/api/files/:filename'],
});

// Handle deprecation with custom deprecation handler that logs and redirects
const handleDeprecatedRoute = (req: IncomingRequest, sock: Socket) => {
  // Create a proper file URL to redirect to
  const filename = req.query?.file;
  const redirectPath = filename ? `/api/files/${filename}` : '/api/files';

  streamLogger.warn('Accessing deprecated streaming route', {
    path: req.path,
    query: req.query,
    clientIp: req.headers['x-forwarded-for'] || sock.remoteAddress,
    replacementRoute: redirectPath,
    timestamp: formatDate(new Date()),
  });

  // Show deprecation notice with 301 redirect
  sendWithContext(
    req,
    sock,
    301,
    {
      'Content-Type': 'text/plain',
      Location: redirectPath,
      'X-Deprecation-Notice': 'This endpoint is deprecated. Please use /api/files routes instead.',
    },
    'This endpoint is deprecated. Redirecting to new API endpoint.',
  );
};

if (config.features.fileStreaming) {
  // GET /stream?file=filename - Streams media files with range support (DEPRECATED)
  router.get('/stream', handleDeprecatedRoute);

  // GET /api/stream?file=filename - Alternative API endpoint (DEPRECATED)
  router.get('/api/stream', handleDeprecatedRoute);

  streamLogger.warn('Deprecated streaming routes registered with redirection', {
    endpoints: ['/stream', '/api/stream'],
    redirectTo: ['/api/files?file=filename', '/api/files/:filename'],
    migrationDeadline: '2025-08-01',
  });
} else {
  streamLogger.warn('File streaming feature is disabled in configuration');
}
