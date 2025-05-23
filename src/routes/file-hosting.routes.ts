/**
 * File Hosting Routes Module
 *
 * Provides routes for the file hosting feature:
 * - GET /api/files - List all available files with pagination and filtering
 * - GET /api/files/search - Search files with advanced filtering
 * - GET /api/files/cache - Get cache statistics or clear cache (admin only)
 * - GET /api/files/stats - Get statistics about files in the system
 * - GET /api/files/:filename - Get a specific file
 * - GET /api/files/:filename/stats - Get detailed statistics for a specific file
 * - POST /api/files - Upload a new file
 * - POST /api/files/bulk - Bulk operations on files
 * - DELETE /api/files/:filename - Delete a file
 * - POST /api/folders - Create a new folder
 * - DELETE /api/folders/:path - Delete a folder
 * - GET /api/folders/:path - List files in a folder
 * - POST /api/files/move - Move a file to a different location
 * - HEAD /api/files/:filename - Get file metadata via headers
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
import { sendWithContext } from '../entities/sendResponse';
import { FileHostingService } from '../modules/file-hosting/fileHostingService';
import path from 'path';

// Create module-specific logger with contextual metadata
const routeLogger = logger.child({
  module: 'routes',
  feature: 'file-hosting',
  component: 'route-handler',
});

// Helper functions for formatting stats
function formatFileSize(sizeInBytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = sizeInBytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(2)} ${units[unitIndex]}`;
}

function formatDuration(durationInSeconds: number): string {
  const hours = Math.floor(durationInSeconds / 3600);
  const minutes = Math.floor((durationInSeconds % 3600) / 60);
  const seconds = Math.floor(durationInSeconds % 60);
  const milliseconds = Math.floor((durationInSeconds % 1) * 1000);

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  } else {
    return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
  }
}

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

      sendWithContext(req, sock, 403, { 'Content-Type': 'text/plain' }, 'Unauthorized access');
      return;
    }

    const action = req.query.action || 'stats';

    if (action === 'clear') {
      const result = fileSvc.clearCache();
      sendWithContext(
        req,
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
      sendWithContext(
        req,
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
  router.get('/api/files', async (req: IncomingRequest, sock: Socket) => {
    const startTime = Date.now();
    routeLogger.info('File listing request started with query params:', req.query);
    try {
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

      routeLogger.debug('Pagination params', { limit, page, sort, order });
      routeLogger.debug(`Starting file query at ${Date.now() - startTime}ms`);

      // Attach updated pagination and sorting back to query
      req.query = { ...req.query, page: page.toString(), limit: limit.toString(), sort, order };

      // Invoke controller and await response handling
      await fileHostingController.listFiles(req, sock);

      routeLogger.info(`File listing completed in ${Date.now() - startTime}ms`);
    } catch (error) {
      routeLogger.error('Error processing file listing', {
        error: error instanceof Error ? error.message : String(error),
      });
      sendWithContext(
        req,
        sock,
        500,
        { 'Content-Type': 'application/json' },
        JSON.stringify({ error: 'Internal server error' }),
      );
    }
  });

  // Add new search endpoint with advanced filtering
  router.get('/api/files/search', (req: IncomingRequest, sock: Socket) => {
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

  // File statistics endpoint - get stats about files in the system
  router.get('/api/files/stats', async (req: IncomingRequest, sock: Socket) => {
    const isAdmin = req.headers['x-admin-key'] === config.adminKey;
    routeLogger.debug('File stats request received', {
      remoteAddress: sock.remoteAddress,
      timestamp: formatDate(new Date()),
      headers: req.headers,
      query: req.query,
    });

    try {
      // Import the stats helper here to avoid circular dependencies
      const { FileHostingStatsHelper } = await import(
        '../modules/file-hosting/fileHostingStatsHelper'
      );
      const statsHelper = new FileHostingStatsHelper(
        path.join(process.cwd(), 'data', 'file_stats.db'),
      );

      // Initialize the helper
      await statsHelper.initialize();

      // Check for query parameters
      const operation = (req.query?.operation as string) || 'aggregate';

      if (operation === 'aggregate') {
        // Get aggregate statistics
        const stats = await statsHelper.getAggregateStats();

        sendWithContext(
          req,
          sock,
          200,
          { 'Content-Type': 'application/json' },
          JSON.stringify({
            success: true,
            stats,
          }),
        );
      } else if (operation === 'query') {
        // Only admin can perform complex queries
        if (!isAdmin) {
          routeLogger.warn('Unauthorized attempt to query file stats', {
            remoteAddress: sock.remoteAddress,
            timestamp: formatDate(new Date()),
          });

          sendWithContext(req, sock, 403, { 'Content-Type': 'text/plain' }, 'Unauthorized access');
          return;
        }

        // Get query parameters
        const queryOptions = {
          mimeType: req.query?.mimeType as string,
          minSize: req.query?.minSize ? parseInt(req.query.minSize as string) : undefined,
          maxSize: req.query?.maxSize ? parseInt(req.query.maxSize as string) : undefined,
          hasAudio: req.query?.hasAudio === 'true',
          hasVideo: req.query?.hasVideo === 'true',
          minWidth: req.query?.minWidth ? parseInt(req.query.minWidth as string) : undefined,
          minHeight: req.query?.minHeight ? parseInt(req.query.minHeight as string) : undefined,
          minDuration: req.query?.minDuration
            ? parseFloat(req.query.minDuration as string)
            : undefined,
          limit: req.query?.limit ? parseInt(req.query.limit as string) : 100,
          offset: req.query?.offset ? parseInt(req.query.offset as string) : 0,
        };

        // Query file stats with filters
        const fileStats = await statsHelper.queryFileStats(queryOptions);

        sendWithContext(
          req,
          sock,
          200,
          { 'Content-Type': 'application/json' },
          JSON.stringify({
            success: true,
            query: queryOptions,
            results: {
              count: fileStats.length,
              items: fileStats,
            },
          }),
        );
      } else if (operation === 'detail' && req.query?.path) {
        // Get stats for a specific file
        const filePath = req.query.path as string;
        const fileStats = await statsHelper.getStatsByPath(filePath);

        if (fileStats) {
          sendWithContext(
            req,
            sock,
            200,
            { 'Content-Type': 'application/json' },
            JSON.stringify({
              success: true,
              stats: fileStats,
            }),
          );
        } else {
          sendWithContext(
            req,
            sock,
            404,
            { 'Content-Type': 'application/json' },
            JSON.stringify({
              success: false,
              message: `No statistics found for file: ${filePath}`,
            }),
          );
        }
      } else {
        // Invalid operation
        sendWithContext(
          req,
          sock,
          400,
          { 'Content-Type': 'application/json' },
          JSON.stringify({
            success: false,
            message: `Invalid operation: ${operation}`,
          }),
        );
      }

      // Close the database connection when done
      await statsHelper.close();
    } catch (err) {
      routeLogger.error('Error processing file stats request', {
        error: (err as Error).message,
        stack: (err as Error).stack,
      });

      sendWithContext(
        req,
        sock,
        500,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: false,
          message: 'Failed to retrieve file statistics: ' + (err as Error).message,
        }),
      );
    }
  });

  // Get detailed file statistics for a specific file
  router.get('/api/files/:filename/stats', async (req: IncomingRequest, sock: Socket) => {
    // Extract filename from URL path
    const pathMatch = req.path.match(/\/api\/files\/([^/]+)\/stats$/);
    const filename = pathMatch ? pathMatch[1] : '';

    routeLogger.debug('File stats detail request received', {
      filename,
      remoteAddress: sock.remoteAddress,
      timestamp: formatDate(new Date()),
      headers: req.headers,
    });

    if (!filename) {
      sendWithContext(
        req,
        sock,
        400,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: false,
          message: 'Missing filename parameter',
        }),
      );
      return;
    }

    try {
      // Import the stats helper
      const { FileHostingStatsHelper } = await import(
        '../modules/file-hosting/fileHostingStatsHelper'
      );
      const statsHelper = new FileHostingStatsHelper(
        path.join(process.cwd(), 'data', 'file_stats.db'),
      );

      // Initialize the helper
      await statsHelper.initialize();

      // First check if the file exists
      try {
        await fileSvc.stat(filename);
      } catch {
        // File doesn't exist
        sendWithContext(
          req,
          sock,
          404,
          { 'Content-Type': 'application/json' },
          JSON.stringify({
            success: false,
            message: `File not found: ${filename}`,
          }),
        );
        await statsHelper.close();
        return;
      }

      // Try to get existing stats from the database
      let fileStats = await statsHelper.getStatsByPath(filename);

      // If stats don't exist or are outdated, collect them now
      if (!fileStats) {
        try {
          const absolutePath = path.join(config.staticDir, filename);
          fileStats = await statsHelper.getFileStats(absolutePath, config.staticDir);
          await statsHelper.saveFileStats(fileStats);

          routeLogger.debug('Generated new file statistics', {
            filename,
            mime: fileStats.mimeType,
          });
        } catch (statsErr) {
          routeLogger.error('Failed to collect file statistics', {
            filename,
            error: (statsErr as Error).message,
          });

          sendWithContext(
            req,
            sock,
            500,
            { 'Content-Type': 'application/json' },
            JSON.stringify({
              success: false,
              message: `Error collecting file statistics: ${(statsErr as Error).message}`,
            }),
          );
          await statsHelper.close();
          return;
        }
      }

      // Add more human-readable versions of some properties
      const enhancedStats = {
        ...fileStats,
        sizeFormatted: formatFileSize(fileStats.size),
        lastModifiedFormatted: formatDate(fileStats.lastModified),
        durationFormatted: fileStats.duration ? formatDuration(fileStats.duration) : undefined,
      };

      // Send the response
      sendWithContext(
        req,
        sock,
        200,
        {
          'Content-Type': 'application/json',
          'Cache-Control': 'max-age=60',
        },
        JSON.stringify({
          success: true,
          fileName: filename,
          stats: enhancedStats,
          _links: {
            self: `/api/files/${encodeURIComponent(filename)}/stats`,
            file: `/api/files/${encodeURIComponent(filename)}`,
            head: {
              href: `/api/files/${encodeURIComponent(filename)}`,
              method: 'HEAD',
              description: 'Get basic file metadata via headers without downloading content',
            },
          },
        }),
      );

      // Close the database connection
      await statsHelper.close();
    } catch (err) {
      routeLogger.error('Error processing file stats detail request', {
        filename,
        error: (err as Error).message,
        stack: (err as Error).stack,
      });

      sendWithContext(
        req,
        sock,
        500,
        { 'Content-Type': 'application/json' },
        JSON.stringify({
          success: false,
          message: 'Failed to retrieve file statistics: ' + (err as Error).message,
        }),
      );
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
    sendWithContext(
      req,
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
        sendWithContext(
          req,
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

      sendWithContext(
        req,
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

      sendWithContext(
        req,
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
        sendWithContext(
          req,
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

      sendWithContext(
        req,
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

      sendWithContext(
        req,
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
      sendWithContext(
        req,
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

      sendWithContext(
        req,
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

      sendWithContext(
        req,
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

      sendWithContext(
        req,
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

      sendWithContext(
        req,
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

  // Get a specific file or file metadata (GET or HEAD)
  router.any('/api/files/:filename', (req: IncomingRequest, sock: Socket) => {
    // Extract filename from URL path using regex instead of req.params
    const pathMatch = req.path.match(/\/api\/files\/([^/]+)$/);
    const filename = pathMatch ? pathMatch[1] : '';

    routeLogger.debug(`${req.method} request received for file`, {
      filename,
      remoteAddress: sock.remoteAddress,
      timestamp: formatDate(new Date()),
      headers: req.headers,
    });

    // Add filename as query parameter for controller compatibility
    req.query = req.query || {};
    req.query.file = filename;

    if (req.method === 'HEAD') {
      fileHostingController.headFile(req, sock);
    } else if (req.method === 'GET') {
      fileHostingController.getFile(req, sock);
    } else {
      // Respond with 405 Method Not Allowed for other methods
      sendWithContext(
        req,
        sock,
        405,
        {
          'Content-Type': 'application/json',
          Allow: 'GET, HEAD, DELETE',
        },
        JSON.stringify({
          success: false,
          message: 'Method not allowed',
          allowedMethods: 'GET, HEAD, DELETE',
          requestedMethod: req.method,
        }),
      );
    }
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
      {
        method: 'GET',
        path: '/api/files/stats',
        description: 'Get statistics about files in the system',
      },
      { method: 'POST', path: '/api/files', description: 'Upload a file' },
      { method: 'POST', path: '/api/files/bulk', description: 'Bulk operations on files' },
      {
        method: 'POST',
        path: '/api/files/move',
        description: 'Move a file to a different location',
      },
      { method: 'GET', path: '/api/files/:filename', description: 'Get file by path parameter' },
      {
        method: 'GET',
        path: '/api/files/:filename/stats',
        description: 'Get detailed statistics for a specific file',
      },
      {
        method: 'HEAD',
        path: '/api/files/:filename',
        description: 'Get file metadata via headers',
      },
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
