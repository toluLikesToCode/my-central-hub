// routes/files.routes.ts
/**
 * @deprecated This file is now obsolete and will be removed in the next major version.
 * All file operations have been moved to file-hosting.routes.ts which handles all /api/files endpoints.
 * Please update your code to use the new routes as follows:
 * - GET /api/files - List all files
 * - GET /api/files?file=filename - Get specific file by query parameter
 * - GET /api/files/:filename - Get specific file by path parameter
 * - POST /api/files - Upload a new file
 * - DELETE /api/files/:filename - Delete a specific file
 */

import router from '../core/router';
import { fileHostingController } from '../modules/file-hosting';
import logger from '../utils/logger';

// Create a module-specific logger with metadata for deprecation warnings
const deprecationLogger = logger.child({
  module: 'files-routes',
  deprecated: true,
});

// Log deprecation warning during route registration
deprecationLogger.warn('Using deprecated /files route', {
  migrationPath: 'Use /api/files routes from file-hosting.routes.ts instead',
  migrationDeadline: '2025-08-01',
});

// Keep for backward compatibility but log deprecation warning on access
router.get('/files', (req, sock) => {
  deprecationLogger.warn('Accessed deprecated /files endpoint', {
    clientIp: req.headers['x-forwarded-for'] || sock.remoteAddress,
    userAgent: req.headers['user-agent'],
    replacementEndpoint: '/api/files',
  });

  // Still handle the request by forwarding to the new controller
  fileHostingController.listFiles(req, sock);
});
