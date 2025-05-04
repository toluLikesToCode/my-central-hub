/* eslint-disable @typescript-eslint/no-unused-vars */
/**
 * src/config/server.config.ts
 * This file contains the server configuration settings.
 */
import dotenv from 'dotenv';
import path, { join } from 'path';
import logger from '../utils/logger';
import process from 'process';

// Using default logger instance imported above

// Load environment variables from .env file
dotenv.config();

export const config = {
  port: process.env.PORT ? parseInt(process.env.PORT, 10) : 8080,
  publicDir: process.env.PUBLIC_DIR
    ? join(process.cwd(), process.env.PUBLIC_DIR)
    : join(process.cwd(), 'public'),
  mediaDir: process.env.MEDIA_DIR
    ? join(process.cwd(), process.env.MEDIA_DIR)
    : join(process.cwd(), 'media'),
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
   * Embedder configuration
   */
  embedding: {
    // Python process settings
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
  testMode: true, // Set to true for testing purposes
};

// Log configuration
if (config.testMode) {
  logger.info(`Server configuration:`);
  logger.info(`- Port: ${config.port}`);
  logger.info(`- Public Directory: ${config.publicDir}`);
  logger.info(`- Media Directory: ${config.mediaDir}`);
  logger.info(`- Header Timeout: ${config.headerTimeoutMs}ms`);
  logger.info(`- Body Timeout: ${config.bodyTimeoutMs}ms`);
  logger.info(`- SQLite DB path: ${config.dbPath}`);
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
}
