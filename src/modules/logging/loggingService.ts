/**
 * Logging Service Module
 *
 * Manages the SQLite database for storing log entries from remote clients.
 * Creates and maintains the database schema, and provides methods to store and retrieve logs.
 *
 * @module modules/logging/loggingService
 */
import sqlite3 from 'sqlite3';
import { open, Database } from 'sqlite';
import { config } from '../../config/server.config';
import path from 'path';
import fs from 'fs';
import {
  Logger,
  FileTransport,
  ConsoleTransport,
  PrettyFormatter,
  JsonFormatter,
} from '../../utils/logger';
import os from 'os';

// Create dedicated logger for the logging service
const loggingServiceLogger = new Logger({
  metadata: {
    module: 'logging',
    component: 'remote-logging-service',
    version: '1.0.0',
    hostname: os.hostname(),
  },
  level: 'debug', // Ensure debug logs are emitted
  transports: [
    new ConsoleTransport({
      formatter: new PrettyFormatter({
        useColors: true,
        showTimestamp: true,
        maxDepth: 3,
        objectKeysLimit: 8,
      }),
      level: 'info',
    }),
    new FileTransport({
      filename: path.join(config.logging.logDir, 'remote-logging-service.log'),
      formatter: new PrettyFormatter({
        useBoxes: false,
        useColors: false,
        showTimestamp: true,
        indent: 2,
        maxDepth: 3,
      }),
      level: 'debug',
    }),
    new FileTransport({
      filename: path.join(config.logging.logDir, 'remote-logging-service.json.log'),
      formatter: new JsonFormatter(),
      level: 'debug',
    }),
  ],
});

// Database connection instance
let db: Database | null = null;
let logsDbPath: string | null = null;

/**
 * Represents a structured log entry in the application's logging system.
 *
 * @interface LogEntry
 *
 * @property {string} level - The severity level of the log (e.g., 'info', 'warn', 'error', 'debug').
 *
 * @property {unknown} message - The main content of the log entry.
 * This can be a string, number, boolean, object, array, or Error instance.
 * When an Error is provided, its message is typically extracted and details are placed in the error property.
 *
 * @property {string} [contextName] - Optional name of the context or component that generated the log.
 *
 * @property {Record<string, unknown>} [meta] - Optional metadata associated with the log entry,
 * providing additional structured information as key-value pairs.
 *
 * @property {string} timestamp - ISO 8601 formatted timestamp indicating when the log was generated.
 *
 * @property {object} [error] - Optional error information when logging exceptions or errors.
 * @property {string} [error.name] - The name of the error (e.g., "TypeError").
 * @property {string} error.message - The error message.
 * @property {string} [error.stack] - The stack trace of the error.
 * @property {unknown} [error.[key: string]] - Additional error properties that might be present.
 */
export interface LogEntry {
  level: string;
  message: unknown;
  contextName?: string;
  context?: string;
  meta?: Record<string, unknown>;
  timestamp: string; // ISO 8601
  error?: {
    name?: string;
    message: string;
    stack?: string;
    [key: string]: unknown;
  };
  url?: string;
  correlationId?: string;
  consoleArgs?: unknown[];
  userAgent?: string;
  ip?: string;
  // Allow any additional fields for flexibility
  [key: string]: unknown;
}

/**
 * Initialize the database connection and schema
 * @param customDbPath Optional custom path for the logs database (for testing)
 */
export async function initializeDatabase(customDbPath?: string): Promise<void> {
  const dbStartTime = Date.now();
  loggingServiceLogger.info('Initializing remote logging database');

  // Use custom path if provided (for tests), else default to data/remote_logs.db
  if (customDbPath) {
    logsDbPath = customDbPath;
  } else {
    const logsDbDir = path.join(process.cwd(), 'data');
    if (!fs.existsSync(logsDbDir)) {
      loggingServiceLogger.info(`Creating database directory`, { path: logsDbDir });
      fs.mkdirSync(logsDbDir, { recursive: true });
    }
    logsDbPath = path.join(logsDbDir, 'remote_logs.db');
  }

  let newDbCreated = false;
  if (!fs.existsSync(logsDbPath)) {
    loggingServiceLogger.info(`Creating new SQLite database file for logs`, { path: logsDbPath });
    fs.closeSync(fs.openSync(logsDbPath, 'w'));
    newDbCreated = true;
  }

  try {
    db = await open({ filename: logsDbPath, driver: sqlite3.Database });

    loggingServiceLogger.debug(`Setting up logs database schema`);
    await db.exec(`
      CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        level TEXT NOT NULL,
        message TEXT NOT NULL,
        contextName TEXT,
        meta TEXT,
        timestamp TEXT NOT NULL,
        error_name TEXT,
        error_message TEXT,
        error_stack TEXT,
        error_details TEXT,
        client_ip TEXT,
        user_agent TEXT,
        url TEXT,
        correlationId TEXT,
        consoleArgs TEXT,
        received_at TEXT NOT NULL,
        raw_entry TEXT
      );
      CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);
      CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
      CREATE INDEX IF NOT EXISTS idx_logs_contextName ON logs(contextName);
      CREATE INDEX IF NOT EXISTS idx_logs_url ON logs(url);
      CREATE INDEX IF NOT EXISTS idx_logs_correlationId ON logs(correlationId);
    `);

    // If newly created DB, log this event prominently
    if (newDbCreated) {
      loggingServiceLogger.success(`Created new SQLite database for logs with schema`, {
        path: logsDbPath,
        tables: ['logs'],
        setup: 'completed',
      });
    }

    const dbSetupTime = Date.now() - dbStartTime;
    loggingServiceLogger.info(`Remote logging database initialization completed`, {
      duration: `${dbSetupTime}ms`,
      status: 'connected',
      newDatabase: newDbCreated,
    });

    return;
  } catch (error) {
    loggingServiceLogger.error('Failed to initialize remote logging database', {
      error:
        error instanceof Error ? { message: error.message, stack: error.stack } : String(error),
      path: logsDbPath,
    });
    throw error;
  }
}

/**
 * Store a log entry in the database
 *
 * @param entry The log entry to store
 * @param clientIp The IP address of the client
 * @param userAgent The user agent of the client
 */
export async function storeLogEntry(
  entry: LogEntry,
  clientIp?: string,
  userAgent?: string,
): Promise<number> {
  if (!db) {
    throw new Error('Database not initialized');
  }

  // Validate required fields
  if (
    !entry ||
    typeof entry.level !== 'string' ||
    typeof entry.timestamp !== 'string' ||
    !('message' in entry)
  ) {
    throw new Error('Log entry must include level, timestamp, and message');
  }

  // Always keep the original entry for raw_entry
  const originalEntry = { ...entry };
  // Use recursive extraction for all relevant fields
  const entryObj = entry as Record<string, unknown>;

  // Helper: always use the deepest value found recursively, skipping 'error' objects for top-level log fields
  function preferRecursive<T>(
    _explicit: T | undefined, // unused
    fieldNames: string[] | string,
    fallback?: T,
    skipKeys?: string[],
  ): T | undefined {
    const found = findFieldDeepest(entryObj, fieldNames, { skipKeys });
    return found !== undefined ? (found.value as T) : fallback;
  }

  // For top-level log fields, skip recursing into 'error' objects
  const level = preferRecursive(entry.level, 'level', 'info', ['error']) as string;
  const messageRaw = preferRecursive(entry.message, 'message', undefined, ['error']);
  const message =
    typeof messageRaw === 'object' ? JSON.stringify(messageRaw) : String(messageRaw ?? '');
  const contextName = preferRecursive(entry.contextName, ['contextName', 'context'], undefined, [
    'error',
  ]) as string | undefined;
  const metaJson = entry.meta ? JSON.stringify(entry.meta) : null;
  const timestamp = preferRecursive(entry.timestamp, 'timestamp', new Date().toISOString(), [
    'error',
  ]) as string;
  const url = preferRecursive(entry.url, 'url', undefined, ['error']) as string | undefined;
  const correlationId = preferRecursive(entry.correlationId, 'correlationId', undefined, [
    'error',
  ]) as string | undefined;
  // Workaround for TypeScript union type assertion issue
  const _consoleArgs = preferRecursive(entry.consoleArgs, 'consoleArgs', undefined, ['error']);
  const consoleArgs = Array.isArray(_consoleArgs) ? _consoleArgs : undefined;
  const _userAgentFinal = preferRecursive(entry.userAgent, 'userAgent', userAgent, ['error']);
  const userAgentFinal = typeof _userAgentFinal === 'string' ? _userAgentFinal : undefined;
  const clientIpFinal = preferRecursive(entry.ip, 'ip', clientIp, ['error']) as string | undefined;

  // Error extraction: find the first error object recursively (now deepest)
  const error =
    entry.error ??
    (findFieldDeepest(entryObj, 'error')?.value as Record<string, unknown> | undefined) ??
    {};
  const errorName =
    error && typeof error === 'object' && 'name' in error ? (error.name as string) : null;
  const errorMessage =
    error && typeof error === 'object' && 'message' in error ? (error.message as string) : null;
  const errorStack =
    error && typeof error === 'object' && 'stack' in error ? (error.stack as string) : null;
  const errorDetails =
    error && typeof error === 'object'
      ? JSON.stringify(
          Object.fromEntries(
            Object.entries(error).filter(([k]) => !['name', 'message', 'stack'].includes(k)),
          ),
        )
      : null;

  // Debug: print extracted values for troubleshooting
  loggingServiceLogger.debug('Extracted log fields', {
    level,
    message,
    contextName,
    timestamp,
    url,
    correlationId,
    consoleArgs,
    userAgentFinal,
    clientIpFinal,
    errorName,
    errorMessage,
    errorStack,
    errorDetails,
  });

  const result = await db.run(
    `INSERT INTO logs (
      level, message, contextName, meta, timestamp,
      error_name, error_message, error_stack, error_details,
      client_ip, user_agent, url, correlationId, consoleArgs, received_at, raw_entry
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      level,
      message,
      contextName,
      metaJson,
      timestamp,
      errorName,
      errorMessage,
      errorStack,
      errorDetails,
      clientIpFinal,
      userAgentFinal,
      url,
      correlationId,
      consoleArgs ? JSON.stringify(consoleArgs) : null,
      new Date().toISOString(),
      JSON.stringify(originalEntry),
    ],
  );
  return result.lastID || 0;
}

/**
 * Get logs from the database with optional filters
 */
export async function getLogs(options: {
  level?: string;
  contextName?: string;
  from?: string;
  to?: string;
  limit?: number;
  offset?: number;
}): Promise<{ logs: unknown[]; total: number }> {
  if (!db) {
    throw new Error('Database not initialized');
  }

  const { level, contextName, from, to, limit = 100, offset = 0 } = options;

  // Build the query conditions
  const conditions: string[] = [];
  const params: unknown[] = [];

  if (level) {
    conditions.push('level = ?');
    params.push(level);
  }

  if (contextName) {
    conditions.push('contextName = ?');
    params.push(contextName);
  }

  if (from) {
    conditions.push('timestamp >= ?');
    params.push(from);
  }

  if (to) {
    conditions.push('timestamp <= ?');
    params.push(to);
  }

  const whereClause = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';

  // Get total count
  const countQuery = `SELECT COUNT(*) as total FROM logs ${whereClause}`;
  const countResult = await db.get(countQuery, params);
  const total = countResult?.total || 0;

  // Get logs with pagination
  const query = `
    SELECT * FROM logs 
    ${whereClause} 
    ORDER BY timestamp DESC, id DESC
    LIMIT ? OFFSET ?
  `;

  const logs = await db.all(query, [...params, limit, offset]);

  // Parse JSON fields
  return {
    logs: logs.map((log) => ({
      ...log,
      meta: log.meta ? JSON.parse(log.meta) : null,
      error_details: log.error_details ? JSON.parse(log.error_details) : null,
      message: tryParseJson(log.message),
    })),
    total,
  };
}

/**
 * Close the database connection
 */
export async function closeDatabase(): Promise<void> {
  if (!db) {
    return;
  }

  try {
    await db.close();
    loggingServiceLogger.success(`SQLite connection closed successfully`);
  } catch (error) {
    loggingServiceLogger.error(`Error closing SQLite connection`, {
      error:
        error instanceof Error ? { message: error.message, stack: error.stack } : String(error),
    });
    throw error;
  } finally {
    db = null;
  }
}

// Helper function to try parsing JSON strings
function tryParseJson(str: string): unknown {
  try {
    return JSON.parse(str);
  } catch {
    return str;
  }
}

// --- Recursive field extraction utility ---
/**
 * Recursively search an object for the deepest occurrence of any of the given field names.
 * Optionally skips recursing into certain keys (e.g., 'error') to avoid extracting log fields from error objects.
 */
function findFieldDeepest(
  obj: unknown,
  fieldNames: string[] | string,
  options?: { depth?: number; maxDepth?: number; seen?: Set<unknown>; skipKeys?: string[] },
): { value: unknown; depth: number } | undefined {
  const fields = Array.isArray(fieldNames) ? fieldNames : [fieldNames];
  const { depth = 0, maxDepth = 16, seen = new Set(), skipKeys = [] } = options || {};
  if (obj == null || typeof obj !== 'object' || seen.has(obj)) return undefined;
  seen.add(obj);
  if (depth > maxDepth) return undefined;
  let found: { value: unknown; depth: number } | undefined = undefined;
  // Recurse into object properties first to ensure deepest match is found before shallow
  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === 'object' && value !== null && !skipKeys.includes(key)) {
      const deeper = findFieldDeepest(value, fields, {
        depth: depth + 1,
        maxDepth,
        seen,
        skipKeys,
      });
      if (deeper && (!found || deeper.depth > found.depth)) {
        found = deeper;
      }
    }
  }
  // Recurse into arrays
  if (Array.isArray(obj)) {
    for (const item of obj) {
      const deeper = findFieldDeepest(item, fields, {
        depth: depth + 1,
        maxDepth,
        seen,
        skipKeys,
      });
      if (deeper && (!found || deeper.depth > found.depth)) {
        found = deeper;
      }
    }
  }
  // Direct match (only update if deeper or not found)
  for (const field of fields) {
    if (Object.prototype.hasOwnProperty.call(obj, field)) {
      if (!found || depth > found.depth) {
        found = { value: (obj as Record<string, unknown>)[field], depth };
      }
    }
  }
  return found;
}

// Initialize on module load
const initPromise = initializeDatabase().catch((err) => {
  loggingServiceLogger.error('Failed to initialize remote logging database', {
    error: err instanceof Error ? { message: err.message, stack: err.stack } : String(err),
  });
});

// Add shutdown handler for graceful closure
process.on('SIGINT', async () => {
  loggingServiceLogger.info('Shutting down remote logging service...');
  await closeDatabase();
});

process.on('SIGTERM', async () => {
  loggingServiceLogger.info('Shutting down remote logging service...');
  await closeDatabase();
});

export { initPromise };
