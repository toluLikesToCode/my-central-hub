/**
 * File Hosting Routes Module
 *
 * Provides routes for the file hosting feature:
 * - GET /api/files - List all available files with pagination and filtering
 * - GET /api/files/search - Search files with advanced filtering
 * - GET /api/files/cache - Get cache statistics or clear cache (admin only)
 * - GET /api/files/:filename - Get a specific file
 * - POST /api/files - Upload a new file
 * - POST /api/files/bulk - Bulk operations on files
 * - DELETE /api/files/:filename - Delete a file
 * - POST /api/folders - Create a new folder
 * - DELETE /api/folders/:path - Delete a folder
 * - GET /api/folders/:path - List files in a folder
 * - POST /api/files/move - Move a file to a different location
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
const fileSvc = new FileHostingService(config.staticDir);

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

  // List all files with pagination and filtering
  router.get('/api/files', (req, sock) => {
    // This is a file listing request
    routeLogger.debug('List files request received', {
      remoteAddress: sock.remoteAddress,
      timestamp: formatDate(new Date()),
      headers: req.headers,
      query: req.query,
    });

    // Process pagination parameters
    const page = parseInt(req.query?.page as string) || 1;
    const limit = parseInt(req.query?.limit as string) || 20;
    const sort = (req.query?.sort as string) || 'name';
    const order = (req.query?.order as string) || 'asc';

    // Add pagination and sorting support to the request
    req.query = {
      ...req.query,
      page: page.toString(),
      limit: limit.toString(),
      sort,
      order,
    };

    fileHostingController.listFiles(req, sock);
  });

  // Add new search endpoint with advanced filtering
  router.get('/api/files/search', (req, sock) => {
    routeLogger.debug('File search request received', {
      remoteAddress: sock.remoteAddress,
      timestamp: formatDate(new Date()),
      headers: req.headers,
      query: req.query,
    });

    // We'll implement this later when we update the controller
    // For now just redirect to listFiles with the search params
    fileHostingController.listFiles(req, sock);
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

  // Bulk operations endpoint
  router.post('/api/files/bulk', (req, sock) => {
    // Parse the request body to extract the operation property safely

    routeLogger.debug('Bulk operation request received', {
      remoteAddress: sock.remoteAddress,
      contentType: req.headers['content-type'],
      timestamp: formatDate(new Date()),
    });

    // This will be implemented later
    // For now return a not implemented response
    sendResponse(
      sock,
      501,
      { 'Content-Type': 'application/json' },
      JSON.stringify({
        success: false,
        message: 'Bulk operations not yet implemented',
      }),
    );
  });

  // Move a file endpoint
  router.post('/api/files/move', async (req, sock) => {
    routeLogger.debug('File move request received', {
      remoteAddress: sock.remoteAddress,
      contentType: req.headers['content-type'],
      timestamp: formatDate(new Date()),
    });

    try {
      // Parse the request body
      let body;
      if (typeof req.body === 'string') {
        body = JSON.parse(req.body);
      } else if (Buffer.isBuffer(req.body)) {
        body = JSON.parse(req.body.toString());
      } else {
        body = req.body;
      }

      const { source, destination } = body;

      if (!source || !destination) {
        sendResponse(
          sock,
          400,
          { 'Content-Type': 'application/json' },
          JSON.stringify({
            success: false,
            message: 'Source and destination paths are required',
          }),
        );
        return;
      }

      await fileSvc.moveFile(source, destination);

      sendResponse(
        sock,
        200,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: true,
          message: 'File moved successfully',
          source,
          destination,
        }),
      );
    } catch (err) {
      routeLogger.error('Error moving file', {
        error: (err as Error).message,
        stack: (err as Error).stack,
      });

      sendResponse(
        sock,
        500,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: false,
          message: 'Failed to move file: ' + (err as Error).message,
        }),
      );
    }
  });

  // Create a new directory
  router.post('/api/folders', async (req, sock) => {
    routeLogger.debug('Create folder request received', {
      remoteAddress: sock.remoteAddress,
      contentType: req.headers['content-type'],
      timestamp: formatDate(new Date()),
    });

    try {
      // Parse the request body
      let body;
      if (typeof req.body === 'string') {
        body = JSON.parse(req.body);
      } else if (Buffer.isBuffer(req.body)) {
        body = JSON.parse(req.body.toString());
      } else {
        body = req.body;
      }

      const { path } = body;

      if (!path) {
        sendResponse(
          sock,
          400,
          { 'Content-Type': 'application/json' },
          JSON.stringify({
            success: false,
            message: 'Folder path is required',
          }),
        );
        return;
      }

      await fileSvc.createDirectory(path);

      sendResponse(
        sock,
        201,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: true,
          message: 'Folder created successfully',
          path,
        }),
      );
    } catch (err) {
      routeLogger.error('Error creating folder', {
        error: (err as Error).message,
        stack: (err as Error).stack,
      });

      sendResponse(
        sock,
        500,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: false,
          message: 'Failed to create folder: ' + (err as Error).message,
        }),
      );
    }
  });

  // Delete a directory
  router.del('/api/folders/:path', async (req, sock) => {
    // Extract path from URL using regex
    const pathMatch = req.path.match(/\/api\/folders\/(.+)$/);
    const folderPath = pathMatch ? decodeURIComponent(pathMatch[1]) : '';

    routeLogger.debug('Delete folder request received', {
      folderPath,
      remoteAddress: sock.remoteAddress,
      timestamp: formatDate(new Date()),
    });

    if (!folderPath) {
      sendResponse(
        sock,
        400,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: false,
          message: 'Folder path is required',
        }),
      );
      return;
    }

    try {
      const recursive = req.query?.recursive === 'true';
      await fileSvc.deleteDirectory(folderPath, recursive);

      sendResponse(
        sock,
        200,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: true,
          message: 'Folder deleted successfully',
          path: folderPath,
          recursive,
        }),
      );
    } catch (err) {
      routeLogger.error('Error deleting folder', {
        error: (err as Error).message,
        folderPath,
      });

      const statusCode = (err as Error).message.includes('not empty') ? 400 : 500;

      sendResponse(
        sock,
        statusCode,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: false,
          message: 'Failed to delete folder: ' + (err as Error).message,
        }),
      );
    }
  });

  // List files in a specific folder
  router.get('/api/folders/:path', async (req, sock) => {
    // Extract path from URL
    const pathMatch = req.path.match(/\/api\/folders\/(.*)$/);
    const folderPath = pathMatch ? decodeURIComponent(pathMatch[1]) : '.';

    routeLogger.debug('List folder contents request received', {
      folderPath,
      remoteAddress: sock.remoteAddress,
      timestamp: formatDate(new Date()),
    });

    try {
      const recursive = req.query?.recursive === 'true';
      const files = await fileSvc.listFiles(folderPath, recursive);

      sendResponse(
        sock,
        200,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: true,
          path: folderPath,
          recursive,
          items: files,
        }),
      );
    } catch (err) {
      routeLogger.error('Error listing folder contents', {
        error: (err as Error).message,
        folderPath,
      });

      sendResponse(
        sock,
        500,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: false,
          message: 'Failed to list folder contents: ' + (err as Error).message,
        }),
      );
    }
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
        path: '/api/files',
        description: 'List files with pagination and filtering',
      },
      {
        method: 'GET',
        path: '/api/files/search',
        description: 'Search files with advanced filtering',
      },
      {
        method: 'GET',
        path: '/api/files/cache',
        description: 'Get cache statistics or clear cache (admin only)',
      },
      { method: 'POST', path: '/api/files', description: 'Upload a file' },
      { method: 'POST', path: '/api/files/bulk', description: 'Bulk operations on files' },
      {
        method: 'POST',
        path: '/api/files/move',
        description: 'Move a file to a different location',
      },
      { method: 'GET', path: '/api/files/:filename', description: 'Get file by path parameter' },
      { method: 'DELETE', path: '/api/files/:filename', description: 'Delete a file' },
      { method: 'POST', path: '/api/folders', description: 'Create a new folder' },
      { method: 'GET', path: '/api/folders/:path', description: 'List files in a folder' },
      { method: 'DELETE', path: '/api/folders/:path', description: 'Delete a folder' },
    ],
    registrationTime: formatDate(new Date()),
  });
} else {
  routeLogger.warn('File hosting feature is disabled in configuration', {
    config: 'config.features.fileHosting',
    enableInstructions: 'Update server.config.ts to enable this feature',
  });
}
