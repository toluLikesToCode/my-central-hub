/**
 * File Hosting Routes Module
 *
 * Provides routes for the file hosting feature:
 * - GET /api/files - List all available files
 * - POST /api/files - Upload a new file
 * - GET /api/files/:filename - Get a specific file
 * - DELETE /api/files/:filename - Delete a file
 *
 * @module routes/file-hosting.routes
 */
import router from '../core/router';
import { fileHostingController } from '../modules/file-hosting';
import { config } from '../config/server.config';
import logger from '../utils/logger';
import { formatDate } from '../utils/dateFormatter';
import { Socket } from 'net';
import { IncomingRequest } from '../entities/http';
import { sendResponse } from '../entities/sendResponse';
import { FileHostingService } from '../modules/file-hosting/fileHostingService';

// Create module-specific logger with contextual metadata
const routeLogger = logger.child({
  module: 'routes',
  feature: 'file-hosting',
  component: 'route-handler',
});

// Record startup time using human-readable format
const startupTime = formatDate(new Date());

// Get a reference to the file service for cache operations
const fileSvc = new FileHostingService(config.mediaDir);

// Log the route registration process with rich contextual data
routeLogger.info('Initializing file hosting routes', {
  enabled: config.features.fileHosting,
  mediaDirectory: config.mediaDir,
  startupTime,
  serverConfig: {
    port: config.port,
    environment: process.env.NODE_ENV || 'development',
  },
});

if (config.features.fileHosting) {
  // Cache management endpoint - admin only
  router.get('/api/files/cache', (req: IncomingRequest, sock: Socket) => {
    const isAdmin = req.headers['x-admin-key'] === config.adminKey;

    if (!isAdmin) {
      routeLogger.warn('Unauthorized cache management attempt', {
        remoteAddress: sock.remoteAddress,
        timestamp: formatDate(new Date()),
        headers: req.headers,
      });

      sendResponse(sock, 403, { 'Content-Type': 'text/plain' }, 'Unauthorized access');
      return;
    }

    const action = req.query.action || 'stats';

    if (action === 'clear') {
      const result = fileSvc.clearCache();
      sendResponse(
        sock,
        200,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: true,
          action: 'clear',
          result,
        }),
      );
    } else {
      // Default: return cache stats
      const stats = fileSvc.getCacheStats();
      sendResponse(
        sock,
        200,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: true,
          stats,
        }),
      );
    }
  });

  // List all files OR get specific file by query parameter
  router.get('/api/files', (req, sock) => {
    // Check if the file query parameter exists - if so, serve the file instead of listing
    if (req.query && req.query.file) {
      // This is a file request via query parameter
      const filename = req.query.file;

      routeLogger.debug('File retrieval request received (query parameter)', {
        filename,
        remoteAddress: sock.remoteAddress,
        timestamp: formatDate(new Date()),
        headers: req.headers,
      });

      fileHostingController.getFile(req, sock);
    } else {
      // This is a file listing request
      routeLogger.debug('List files request received', {
        remoteAddress: sock.remoteAddress,
        timestamp: formatDate(new Date()),
        headers: req.headers,
      });

      fileHostingController.listFiles(req, sock);
    }
  });

  // Upload a file
  router.post('/api/files', (req, sock) => {
    routeLogger.debug('File upload request received', {
      remoteAddress: sock.remoteAddress,
      contentType: req.headers['content-type'],
      contentLength: req.headers['content-length'],
      timestamp: formatDate(new Date()),
    });

    fileHostingController.uploadFile(req, sock);
  });

  // Get a specific file - Extract filename from path
  router.get('/api/files/:filename', (req, sock) => {
    // Extract filename from URL path using regex instead of req.params
    const pathMatch = req.path.match(/\/api\/files\/([^/]+)$/);
    const filename = pathMatch ? pathMatch[1] : '';

    routeLogger.debug('File retrieval request received (path parameter)', {
      filename,
      remoteAddress: sock.remoteAddress,
      timestamp: formatDate(new Date()),
      headers: req.headers,
    });

    // Add filename as query parameter for controller compatibility
    req.query = req.query || {};
    req.query.file = filename;

    fileHostingController.getFile(req, sock);
  });

  // Delete a specific file - Extract filename from path
  router.del('/api/files/:filename', (req, sock) => {
    // Extract filename from URL path using regex instead of req.params
    const pathMatch = req.path.match(/\/api\/files\/([^/]+)$/);
    const filename = pathMatch ? pathMatch[1] : '';

    routeLogger.debug('File deletion request received', {
      filename,
      remoteAddress: sock.remoteAddress,
      timestamp: formatDate(new Date()),
      headers: req.headers,
    });

    // Add filename as query parameter for controller compatibility
    req.query = req.query || {};
    req.query.file = filename;

    fileHostingController.deleteFile(req, sock);
  });

  routeLogger.success('File hosting routes registered successfully', {
    routes: [
      {
        method: 'GET',
        path: '/api/files/cache',
        description: 'Get cache statistics or clear cache (admin only)',
      },
      {
        method: 'GET',
        path: '/api/files',
        description: 'List files or get file by query parameter',
      },
      { method: 'POST', path: '/api/files', description: 'Upload a file' },
      { method: 'GET', path: '/api/files/:filename', description: 'Get file by path parameter' },
      { method: 'DELETE', path: '/api/files/:filename', description: 'Delete a file' },
    ],
    registrationTime: formatDate(new Date()),
  });
} else {
  routeLogger.warn('File hosting feature is disabled in configuration', {
    config: 'config.features.fileHosting',
    enableInstructions: 'Update server.config.ts to enable this feature',
  });
}
