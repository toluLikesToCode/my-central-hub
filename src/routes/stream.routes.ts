/**
 * Stream Routes Module
 *
 * Handles media file streaming functionality with range support
 * for video and audio content delivery with optimal performance.
 *
 * @module routes/stream.routes
 */
import router from '../core/router';
import { fileStreamingController } from '../modules/file-streaming';
import { config } from '../config/server.config';
import logger from '../utils/logger';

// Create a module-specific logger with metadata
const streamLogger = logger.child({
  module: 'stream-routes',
  feature: 'media-streaming',
});

// Log when stream routes are registered during startup
streamLogger.info('Registering streaming routes', {
  enabled: config.features.fileStreaming,
  mediaDirectory: config.mediaDir,
});

if (config.features.fileStreaming) {
  // GET /stream?file=filename - Streams media files with range support
  router.get('/stream', fileStreamingController.handleStream);

  // GET /api/stream?file=filename - Alternative API endpoint
  router.get('/api/stream', fileStreamingController.handleStream);

  streamLogger.success('Streaming routes registered successfully', {
    endpoints: ['/stream', '/api/stream'],
  });
} else {
  streamLogger.warn('File streaming feature is disabled in configuration');
}
