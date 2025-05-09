/* eslint-disable @typescript-eslint/no-unused-vars */
/**
 * src/config/server.config.ts
 * This file contains the server configuration settings.
 */
import dotenv from 'dotenv';
import path, { join } from 'path';
import logger from '../utils/logger';
import process from 'process';
import { DateTimeConfig } from '../utils/dateFormatter';

// Load environment variables from .env file
dotenv.config();

export const config = {
  port: process.env.PORT ? parseInt(process.env.PORT, 10) : 8080,
  publicDir: process.env.PUBLIC_DIR
    ? join(process.cwd(), process.env.PUBLIC_DIR)
    : join(process.cwd(), 'public'),
  mediaDir: process.env.MEDIA_DIR
    ? join(process.cwd(), process.env.MEDIA_DIR)
    : join(process.cwd(), 'public', 'media'),
  headerTimeoutMs: process.env.HEADER_TIMEOUT_MS
    ? Math.max(parseInt(process.env.HEADER_TIMEOUT_MS, 10), 0)
    : 10000,
  bodyTimeoutMs: process.env.BODY_TIMEOUT_MS
    ? Math.max(parseInt(process.env.BODY_TIMEOUT_MS, 10), 0)
    : 15000,
  /**
   * Path to the SQLite database file. Will be created if missing.
   */
  dbPath: process.env.DB_PATH ? process.env.DB_PATH : join(process.cwd(), 'data', 'metrics.db'),
  /**
   * Admin key for privileged operations (cache management, etc.)
   */
  adminKey: process.env.ADMIN_KEY || 'admin-secret-key',
  /**
   * File caching configuration
   */
  fileCache: {
    enabled: process.env.FILE_CACHE_ENABLED !== 'false', // Default to true
    maxSize: parseInt(process.env.FILE_CACHE_MAX_SIZE || '209715200', 10), // 200MB default
    maxAge: parseInt(process.env.FILE_CACHE_MAX_AGE || '600000', 10), // 10 minutes default
  },
  /**
   * Feature toggles for modularity and configurability
   */
  features: {
    metrics: true,
    fileHosting: true,
    fileStreaming: true,
    embeddingService: true,
    // Add more features here as needed
  },

  /**
   * Logging configuration
   */
  logging: {
    level: process.env.LOG_LEVEL || 'info', // Default log level
    format: process.env.LOG_FORMAT || 'json', // Default log format
    logDir: process.env.LOG_DIR || join(process.cwd(), 'logs'), // Centralized log directory
  },
  /**
   * Date and time formatting configuration
   * Now using DateTimeConfig imported from dateFormatter.ts
   */
  dateTime: DateTimeConfig,

  /**
   * Embedder configuration
   */
  embedding: {
    // Python process settings
    maxRetries: 3,
    retryDelayMs: 1000,
    timeoutMs: 30000,
    // Embedding service settings
    serviceUrl: process.env.EMBEDDING_SERVICE_URL || 'http://192.168.1.107:3456',
    pythonExecutable: process.env.PYTHON_EXECUTABLE || 'python3',
    pythonScriptPath:
      process.env.PYTHON_SCRIPT_PATH ||
      path.resolve(process.cwd(), 'python', 'embedding_service_helper.py'), // Path relative to project root
    pythonLogPath: process.env.PYTHON_LOG_PATH, // Optional: Path for python script's own log, defaults to alongside script if not set
    // Model/Processing Args passed to Python script
    modelArgs: [
      '--enable_augmentation',
      '--log',
      '--debug',
      '-n',
      '30',
      // Example: '--model', 'openai/clip-vit-base-patch32' is now expected to be set here
    ],
    defaultModel: 'openai/clip-vit-base-patch32', // Default model if not in args
    defaultNumFrames: 15,
    enableAugmentation: false, // Default augmentation flag for python script
    // Service behavior
    inactivityTimeoutMs: 10 * 60 * 1000,
    scriptTimeoutMs: 30 * 60 * 1000,
    debug: true,
    log: true,
    inputDir: process.env.EMBED_DIR,
  },
  testMode: false, // Set to true for testing purposes
  staticDir: process.env.STATIC_DIR || join(process.cwd(), 'static'), // Static files directory
};

// Only log configuration if logger is defined and we're in test mode
if (logger && config.testMode) {
  // Wrap in try/catch to avoid potential startup issues
  try {
    logger.info(`Server configuration:`);
    logger.info(`- Port: ${config.port}`);
    logger.info(`- Public Directory: ${config.publicDir}`);
    logger.info(`- Media Directory: ${config.mediaDir}`);
    logger.info(`- Log Directory: ${config.logging.logDir}`);
    logger.info(`- Static Directory: ${config.staticDir}`);
    logger.info(`- Header Timeout: ${config.headerTimeoutMs}ms`);
    logger.info(`- Body Timeout: ${config.bodyTimeoutMs}ms`);
    logger.info(`- SQLite DB path: ${config.dbPath}`);
    logger.info(`- Admin Key: ${config.adminKey}`);
    logger.info(`- File Cache Enabled: ${config.fileCache.enabled}`);
    logger.info(`- File Cache Max Size: ${config.fileCache.maxSize} bytes`);
    logger.info(`- File Cache Max Age: ${config.fileCache.maxAge}ms`);
    logger.info(`- Active Features:`, {
      features: Object.entries(config.features)
        .filter(([_, enabled]) => enabled)
        .map(([feature]) => feature),
    });
    logger.info(`- Log Level: ${config.logging.level}`);
    logger.info(`- Python Executable: ${config.embedding.pythonExecutable}`);
    logger.info(`- Python Script Path: ${config.embedding.pythonScriptPath}`);
    logger.info(`- Embedding Inactivity Timeout: ${config.embedding.inactivityTimeoutMs}ms`);
    logger.info(`- Embedding Script Timeout: ${config.embedding.scriptTimeoutMs}ms`);
    logger.info(`- Active Model Args: ${config.embedding.modelArgs.join(', ')}`);
    logger.info(`- Timezone: ${config.dateTime.timezone}`);
    logger.info(`- Date Format: ${config.dateTime.format}`);
  } catch (error) {
    // In case of any logging error during startup, log to console instead
    console.warn('Error logging configuration:', error);
  }
}
