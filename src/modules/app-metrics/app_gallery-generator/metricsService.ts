import { z } from 'zod';
import sqlite3 from 'sqlite3';
import { open, Database } from 'sqlite';
import { config } from '../../../config/server.config';
import {
  Logger,
  FileTransport,
  ConsoleTransport,
  PrettyFormatter,
  JsonFormatter,
} from '../../../utils/logger';
import { formatDate } from '../../../utils/dateFormatter';
import fs from 'fs';
import path from 'path';
import os from 'os';

// Add at top of file after existing imports
let shutdownHandlersRegistered = false;

// Create dedicated logger for metrics service with comprehensive performance tracking
const metricsServiceLogger = new Logger({
  metadata: {
    module: 'app-metrics',
    component: 'gallery-generator-service',
    version: '2.0.0',
    hostname: os.hostname(),
  },
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
      filename: path.join(config.logging.logDir, 'metricsService.log'),
      formatter: new PrettyFormatter({
        useBoxes: false,
        useColors: false,
        showTimestamp: true,
        indent: 3,
        arrayLengthLimit: 15,
        objectKeysLimit: 10,
        maxDepth: 4,
        stringLengthLimit: 300,
      }),
      level: 'debug',
    }),
    // Add JSON-formatted logs for data analysis
    new FileTransport({
      filename: path.join(config.logging.logDir, 'metricsService.json.log'),
      formatter: new JsonFormatter(),
      level: 'debug',
    }),
  ],
});

// --- Zod Schemas ---
const PerfLogEntrySchema = z
  .object({
    timestamp: z.string(),
    perfNow: z.number(),
    memory: z
      .object({
        usedJSHeapSize: z.number().optional(),
        totalJSHeapSize: z.number().optional(),
        jsHeapSizeLimit: z.number().optional(),
      })
      .optional()
      .nullable(), // allow null as well as undefined
    action: z.string(),
    sessionId: z.string().optional(),
    batchId: z.number().optional(),
    uploadMode: z.string().optional(),
  })
  .catchall(z.unknown());

const EngagementStatsSchema = z.object({
  views: z.number(),
  lastViewedAt: z.number(),
  totalWatchMs: z.number(),
  completions: z.number(),
});
const EngagementMapSchema = z.record(EngagementStatsSchema);

const DebugLogSchema = z.object({
  message: z.string(),
  timestamp: z.string(),
});
const DebugLogsSchema = z.array(DebugLogSchema);

const SessionMetricsSchema = z.object({
  sessionId: z.string(),
  startTime: z.number(),
  firstClickTime: z.number().optional(),
  scrollDistance: z.number(),
  itemsLoaded: z.number(),
  infiniteScrollLoads: z.number(),
  hoverThumbnails: z.number(),
  gridClickOpenCount: z.number(),
  modalsOpened: z.number(),
  modalTotalTime: z.number(),
  carouselNavigationCount: z.number(),
  modalContentCounts: z.object({
    video: z.number(),
    image: z.number(),
  }),
  videoMetrics: z.object({
    plays: z.number(),
    completions: z.number(),
    watchTime: z.number(),
    manualStarts: z.number(),
    autoPlays: z.number(),
  }),
  performanceMetrics: z.object({
    preloadDurations: z.array(z.number()),
    longTasks: z.number(),
    infiniteLoadTimes: z.array(z.number()),
    modalAnimationLatencies: z.array(z.number()),
  }),
});

const FullPayloadSchema = z
  .object({
    engagement: EngagementMapSchema,
    perfLog: z.array(PerfLogEntrySchema),
    debug: DebugLogsSchema,
    timestamp: z.string(),
    sessionStart: z.number(),
    sessionMetrics: SessionMetricsSchema,
  })
  .catchall(z.unknown());

export type PerfLogEntry = z.infer<typeof PerfLogEntrySchema>;
export type FullPayload = z.infer<typeof FullPayloadSchema>;

// Export metricsPayloadSchema for controller usage
export const metricsPayloadSchema = FullPayloadSchema;

// Type guard for PerfLogEntry array
export function isPerfLogArray(arr: unknown): arr is PerfLogEntry[] {
  return Array.isArray(arr) && arr.every((item) => PerfLogEntrySchema.safeParse(item).success);
}

// --- DB Setup ---
let db: Database | null = null;
export async function initDb(): Promise<Database> {
  const dbStartTime = Date.now();
  metricsServiceLogger.info(`Initializing database connection`, {
    dbPath: config.dbPath,
    initializing: !db,
  });

  if (db) return db;

  // Compute the directory that should contain the .db file:
  const dbDir = path.dirname(config.dbPath);
  if (!fs.existsSync(dbDir)) {
    metricsServiceLogger.info(`Creating database directory`, { path: dbDir });
    fs.mkdirSync(dbDir, { recursive: true });
  }

  let newDbCreated = false;
  if (!fs.existsSync(config.dbPath)) {
    metricsServiceLogger.info(`Creating new SQLite database file`, { path: config.dbPath });
    fs.closeSync(fs.openSync(config.dbPath, 'w'));
    newDbCreated = true;
  }

  try {
    db = await open({ filename: config.dbPath, driver: sqlite3.Database });

    metricsServiceLogger.debug(`Setting up database schema`);
    await db.exec(`
      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        sessionStart INTEGER
      );

      CREATE TABLE IF NOT EXISTS engagement (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sessionId TEXT,
        itemId TEXT,
        views INTEGER,
        lastViewedAt INTEGER,
        totalWatchMs INTEGER,
        completions INTEGER,
        FOREIGN KEY(sessionId) REFERENCES sessions(id)
      );

      CREATE TABLE IF NOT EXISTS perfLog (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sessionId TEXT,
        timestamp TEXT,
        perfNow REAL,
        usedJSHeapSize INTEGER,
        totalJSHeapSize INTEGER,
        jsHeapSizeLimit INTEGER,
        action TEXT,
        batchId INTEGER,
        uploadMode TEXT,
        details TEXT,
        FOREIGN KEY(sessionId) REFERENCES sessions(id)
      );

      CREATE TABLE IF NOT EXISTS debug (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sessionId TEXT,
        message TEXT,
        timestamp TEXT,
        FOREIGN KEY(sessionId) REFERENCES sessions(id)
      );

      CREATE TABLE IF NOT EXISTS sessionMetrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sessionId TEXT,
        data TEXT,
        FOREIGN KEY(sessionId) REFERENCES sessions(id)
      );

      -- Add indexes for performance
      CREATE INDEX IF NOT EXISTS idx_perfLog_sessionId ON perfLog(sessionId);
      CREATE INDEX IF NOT EXISTS idx_perfLog_action ON perfLog(action);
      CREATE INDEX IF NOT EXISTS idx_engagement_sessionId ON engagement(sessionId);
      CREATE INDEX IF NOT EXISTS idx_debug_sessionId ON debug(sessionId);
    `);

    // If newly created DB, log this event prominently
    if (newDbCreated) {
      metricsServiceLogger.success(`Created new SQLite database with schema`, {
        path: config.dbPath,
        tables: ['sessions', 'engagement', 'perfLog', 'debug', 'sessionMetrics'],
        setup: 'completed',
      });
    }

    const dbSetupTime = Date.now() - dbStartTime;
    metricsServiceLogger.info(`Database initialization completed`, {
      duration: `${dbSetupTime}ms`,
      status: 'connected',
      newDatabase: newDbCreated,
    });

    // Register shutdown handlers once after DB is ready
    if (!shutdownHandlersRegistered) {
      shutdownHandlersRegistered = true;
      process.on('SIGINT', async () => {
        metricsServiceLogger.info(`SIGINT received. Initiating graceful shutdown`, {
          event: 'SIGINT',
        });
        try {
          await db?.close();
          metricsServiceLogger.success(`SQLite connection closed successfully`, {
            event: 'SIGINT',
          });
        } catch (err) {
          metricsServiceLogger.error(`Error closing SQLite connection`, {
            error: (err as Error).message,
            event: 'SIGINT',
          });
        }
        process.exit(0);
      });
      process.on('SIGTERM', async () => {
        metricsServiceLogger.info(`SIGTERM received. Initiating graceful shutdown`, {
          event: 'SIGTERM',
        });
        try {
          await db?.close();
          metricsServiceLogger.success(`SQLite connection closed successfully`, {
            event: 'SIGTERM',
          });
        } catch (err) {
          metricsServiceLogger.error(`Error closing SQLite connection`, {
            error: (err as Error).message,
            event: 'SIGTERM',
          });
        }
        process.exit(0);
      });
    }

    return db;
  } catch (err) {
    const error = err as Error;
    metricsServiceLogger.error(`Database initialization failed`, {
      error: error.message,
      stack: error.stack,
      dbPath: config.dbPath,
    });
    throw err;
  }
}

// --- Performance monitoring for database operations ---
async function monitorDbOperation<T>(
  operation: string,
  sessionId: string | undefined,
  fn: () => Promise<T>,
): Promise<T> {
  const startTime = performance.now();
  try {
    const result = await fn();
    const duration = performance.now() - startTime;

    // Log performance data for DB operations
    if (duration > 500) {
      // Log slower operations with warning level
      metricsServiceLogger.warn(`Slow database operation: ${operation}`, {
        operation,
        sessionId: sessionId || 'unknown',
        duration: `${duration.toFixed(2)}ms`,
        timestamp: formatDate(new Date()),
      });
    } else if (process.env.LOG_DB_PERF === 'true') {
      // Log all operations in detailed performance tracking mode
      metricsServiceLogger.debug(`Database operation: ${operation}`, {
        operation,
        sessionId: sessionId || 'unknown',
        duration: `${duration.toFixed(2)}ms`,
      });
    }

    return result;
  } catch (err) {
    const error = err as Error;
    const duration = performance.now() - startTime;

    metricsServiceLogger.error(`Database operation failed: ${operation}`, {
      operation,
      sessionId: sessionId || 'unknown',
      duration: `${duration.toFixed(2)}ms`,
      error: error.message,
      stack: error.stack,
    });

    throw err;
  }
}

// --- Utility: Accept and log all valid data, never reject whole request ---
function logInvalid(type: string, data: unknown, error: unknown) {
  metricsServiceLogger.warn(`Invalid metrics entry detected`, {
    entryType: type,
    validationError: error,
    dataPreview: JSON.stringify(data).substring(0, 150) + '...',
  });
}

// --- Session Helper ---
async function ensureSessionExists(
  database: Database,
  sessionId: string,
  timestamp?: string,
  sessionStart?: number,
): Promise<void> {
  return monitorDbOperation('ensureSessionExists', sessionId, async () => {
    // Try to find session by id
    const sessionRow = await database.get('SELECT id FROM sessions WHERE id = ?', sessionId);
    if (!sessionRow) {
      const currentTime = formatDate(new Date());
      await database.run(
        'INSERT INTO sessions (id, timestamp, sessionStart) VALUES (?, ?, ?)',
        sessionId,
        timestamp || currentTime,
        sessionStart ?? Date.now(),
      );

      metricsServiceLogger.info(`Created new session record`, {
        sessionId,
        timestamp: timestamp || currentTime,
        sessionStart: sessionStart ? formatDate(new Date(sessionStart)) : 'not provided',
        created: currentTime,
      });
    }
  });
}

export async function saveMetrics(
  payload: unknown,
  clientSessionId?: string,
  env?: unknown,
): Promise<void> {
  const saveStart = Date.now();
  const dbOperations: string[] = [];
  const metrics = {
    payloadType: Array.isArray(payload) ? 'PerfLogBatch' : typeof payload,
    clientSessionId,
    received: formatDate(new Date()),
    batchSize: Array.isArray(payload) ? payload.length : 1,
    environment:
      typeof env === 'object' && env !== null ? JSON.stringify(env).substring(0, 100) : 'default',
  };

  metricsServiceLogger.info(`Saving metrics data`, metrics);

  try {
    const database = await initDb();

    // PerfLog batch
    if (Array.isArray(payload)) {
      metricsServiceLogger.debug(`Processing performance log batch`, {
        batchSize: payload.length,
        sessionId: clientSessionId || 'unknown',
      });

      let validEntries = 0;
      let invalidEntries = 0;

      for (const entry of payload) {
        const result = PerfLogEntrySchema.safeParse(entry);
        if (!result.success) {
          logInvalid('perfLog', entry, result.error.format());
          invalidEntries++;
          continue;
        }

        // Resolve sessionId
        const resolvedSessionId = result.data.sessionId || clientSessionId;
        if (!resolvedSessionId) {
          metricsServiceLogger.warn(`No sessionId found for perfLog entry`, {
            entry: JSON.stringify(entry).substring(0, 200),
          });
          invalidEntries++;
          continue;
        }

        await ensureSessionExists(database, resolvedSessionId, result.data.timestamp);

        const { timestamp, perfNow, memory, action, batchId, uploadMode, ...rest } = result.data;
        const details = JSON.stringify({ ...rest });

        await monitorDbOperation('insertPerfLog', resolvedSessionId, async () => {
          await database.run(
            `INSERT INTO perfLog (sessionId, timestamp, perfNow, usedJSHeapSize, totalJSHeapSize, jsHeapSizeLimit, action, batchId, uploadMode, details)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
            resolvedSessionId,
            timestamp,
            perfNow,
            memory?.usedJSHeapSize ?? -1,
            memory?.totalJSHeapSize ?? -1,
            memory?.jsHeapSizeLimit ?? -1,
            action,
            batchId ?? null,
            uploadMode ?? null,
            details,
          );
        });

        dbOperations.push('insertPerfLog');
        validEntries++;
      }

      const processingTime = Date.now() - saveStart;
      metricsServiceLogger.success(`Performance log batch processing completed`, {
        totalEntries: payload.length,
        validEntries,
        invalidEntries,
        processingTime: `${processingTime}ms`,
        throughput: `${Math.round(validEntries / (processingTime / 1000))} items/sec`,
        sessionId: clientSessionId || 'unknown',
      });

      return;
    }

    // Full payload processing
    const result = FullPayloadSchema.safeParse(payload);
    if (!result.success) {
      logInvalid('fullPayload', payload, result.error.format());

      // Attempt to salvage valid subfields
      if (payload && typeof payload === 'object') {
        metricsServiceLogger.info(`Attempting to salvage valid data from invalid payload`);
        let salvaged = 0;

        // Try perfLog
        const perfLogArray = Array.isArray((payload as Record<string, unknown>).perfLog)
          ? ((payload as Record<string, unknown>).perfLog as unknown[])
          : [];

        if (perfLogArray.length > 0) {
          metricsServiceLogger.debug(`Salvaging perfLog entries`, {
            count: perfLogArray.length,
          });
        }

        for (const entry of perfLogArray) {
          const perfResult = PerfLogEntrySchema.safeParse(entry);
          if (perfResult.success) {
            const { timestamp, perfNow, memory, action, batchId, uploadMode, ...rest } =
              perfResult.data;
            const details = JSON.stringify({ ...rest });

            await monitorDbOperation('insertSalvagedPerfLog', clientSessionId, async () => {
              await database.run(
                `INSERT INTO perfLog (sessionId, timestamp, perfNow, usedJSHeapSize, totalJSHeapSize, jsHeapSizeLimit, action, batchId, uploadMode, details)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                clientSessionId || null,
                timestamp,
                perfNow,
                memory?.usedJSHeapSize ?? -1,
                memory?.totalJSHeapSize ?? -1,
                memory?.jsHeapSizeLimit ?? -1,
                action,
                batchId ?? null,
                uploadMode ?? null,
                details,
              );
            });

            dbOperations.push('insertSalvagedPerfLog');
            salvaged++;
          } else {
            logInvalid('perfLog', entry, perfResult.error.format());
          }
        }

        // Try debug
        const debugArray = Array.isArray((payload as Record<string, unknown>).debug)
          ? ((payload as Record<string, unknown>).debug as unknown[])
          : [];

        if (debugArray.length > 0) {
          metricsServiceLogger.debug(`Salvaging debug logs`, {
            count: debugArray.length,
          });
        }

        for (const entry of debugArray) {
          const debugResult = DebugLogSchema.safeParse(entry);
          if (debugResult.success) {
            await monitorDbOperation('insertSalvagedDebug', clientSessionId, async () => {
              await database.run(
                `INSERT INTO debug (sessionId, message, timestamp) VALUES (?, ?, ?)`,
                clientSessionId || null,
                debugResult.data.message,
                debugResult.data.timestamp,
              );
            });

            dbOperations.push('insertSalvagedDebug');
            salvaged++;
          } else {
            logInvalid('debug', entry, debugResult.error.format());
          }
        }

        // Try engagement
        const engagementObj = (payload as Record<string, unknown>).engagement;
        if (engagementObj && typeof engagementObj === 'object' && !Array.isArray(engagementObj)) {
          const engagementKeys = Object.keys(engagementObj);

          if (engagementKeys.length > 0) {
            metricsServiceLogger.debug(`Salvaging engagement data`, {
              itemCount: engagementKeys.length,
            });
          }

          for (const [itemId, stats] of Object.entries(engagementObj)) {
            const statsResult = EngagementStatsSchema.safeParse(stats);
            if (statsResult.success) {
              await monitorDbOperation('insertSalvagedEngagement', clientSessionId, async () => {
                await database.run(
                  `INSERT INTO engagement (sessionId, itemId, views, lastViewedAt, totalWatchMs, completions) VALUES (?, ?, ?, ?, ?, ?)`,
                  clientSessionId || null,
                  itemId,
                  statsResult.data.views,
                  statsResult.data.lastViewedAt,
                  statsResult.data.totalWatchMs,
                  statsResult.data.completions,
                );
              });

              dbOperations.push('insertSalvagedEngagement');
              salvaged++;
            } else {
              logInvalid('engagement', stats, statsResult.error.format());
            }
          }
        }

        if (salvaged > 0) {
          metricsServiceLogger.info(`Salvaged ${salvaged} valid entries from invalid payload`, {
            processingTime: `${Date.now() - saveStart}ms`,
          });
        }
      }
      return;
    }

    // If we get here, we have a valid full payload
    const {
      engagement,
      perfLog,
      debug,
      timestamp,
      sessionStart,
      sessionMetrics: sessionMetricsFromPayload,
    } = result.data;

    // Resolve sessionId
    const resolvedSessionId = sessionMetricsFromPayload.sessionId || clientSessionId;
    const displaySessionId = resolvedSessionId || 'unknown-session';

    metricsServiceLogger.info(`Processing validated full metrics payload`, {
      sessionId: displaySessionId,
      engagementCount: Object.keys(engagement).length,
      perfLogCount: perfLog.length,
      debugCount: debug.length,
      startTime: sessionStart ? formatDate(new Date(sessionStart)) : 'unknown',
    });

    await ensureSessionExists(database, displaySessionId, timestamp, sessionStart);

    // Process engagement data
    if (Object.keys(engagement).length > 0) {
      metricsServiceLogger.debug(`Processing engagement data`, {
        itemCount: Object.keys(engagement).length,
        sessionId: displaySessionId,
      });

      for (const [itemId, stats] of Object.entries(engagement)) {
        await monitorDbOperation('insertEngagement', displaySessionId, async () => {
          await database.run(
            `INSERT INTO engagement (sessionId, itemId, views, lastViewedAt, totalWatchMs, completions) VALUES (?, ?, ?, ?, ?, ?)`,
            displaySessionId,
            itemId,
            stats.views,
            stats.lastViewedAt,
            stats.totalWatchMs,
            stats.completions,
          );
        });

        dbOperations.push('insertEngagement');
      }
    }

    // Process perfLog entries
    if (perfLog.length > 0) {
      metricsServiceLogger.debug(`Processing performance log entries`, {
        count: perfLog.length,
        sessionId: displaySessionId,
      });

      for (const entry of perfLog) {
        const { timestamp, perfNow, memory, action, batchId, uploadMode, ...rest } = entry;
        const details = JSON.stringify({ ...rest });

        await monitorDbOperation('insertPerfLog', displaySessionId, async () => {
          await database.run(
            `INSERT INTO perfLog (sessionId, timestamp, perfNow, usedJSHeapSize, totalJSHeapSize, jsHeapSizeLimit, action, batchId, uploadMode, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
            displaySessionId,
            timestamp,
            perfNow,
            memory?.usedJSHeapSize ?? -1,
            memory?.totalJSHeapSize ?? -1,
            memory?.jsHeapSizeLimit ?? -1,
            action,
            batchId ?? null,
            uploadMode ?? null,
            details,
          );
        });

        dbOperations.push('insertPerfLog');
      }
    }

    // Process debug entries
    if (debug.length > 0) {
      metricsServiceLogger.debug(`Processing debug log entries`, {
        count: debug.length,
        sessionId: displaySessionId,
      });

      for (const entry of debug) {
        await monitorDbOperation('insertDebug', displaySessionId, async () => {
          await database.run(
            `INSERT INTO debug (sessionId, message, timestamp) VALUES (?, ?, ?)`,
            displaySessionId,
            entry.message,
            entry.timestamp,
          );
        });

        dbOperations.push('insertDebug');
      }
    }

    // Insert sessionMetrics as JSON
    metricsServiceLogger.debug(`Processing session metrics`, {
      sessionId: displaySessionId,
    });

    await monitorDbOperation('insertSessionMetrics', displaySessionId, async () => {
      await database.run(
        `INSERT INTO sessionMetrics (sessionId, data) VALUES (?, ?)`,
        displaySessionId,
        JSON.stringify(sessionMetricsFromPayload),
      );
    });

    dbOperations.push('insertSessionMetrics');

    // Log operation summary
    const operationCounts = dbOperations.reduce(
      (acc, op) => {
        acc[op] = (acc[op] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>,
    );

    const processingTime = Date.now() - saveStart;

    // Calculate operations per second
    const totalOps = dbOperations.length;
    const opsPerSecond = Math.round(totalOps / (processingTime / 1000));

    metricsServiceLogger.success(`Metrics data saved successfully`, {
      sessionId: displaySessionId,
      processingTime: `${processingTime}ms`,
      operationCounts,
      totalOperations: totalOps,
      throughput: `${opsPerSecond} ops/sec`,
      timestamp: formatDate(new Date()),
    });
  } catch (err) {
    const error = err as Error;
    metricsServiceLogger.error(`Failed to save metrics data`, {
      error: error.message,
      stack: error.stack,
      sessionId: clientSessionId || 'unknown',
      processingTime: `${Date.now() - saveStart}ms`,
      payload:
        typeof payload === 'object'
          ? JSON.stringify(payload).substring(0, 200) + '...'
          : String(payload),
    });

    throw error;
  }
}

/**
 * Retrieves recent session metrics for troubleshooting and monitoring
 *
 * @param limit - Maximum number of sessions to retrieve
 * @returns Promise with recent session data
 */
export async function getRecentSessions(limit = 20): Promise<unknown[]> {
  const queryStart = Date.now();

  metricsServiceLogger.info(`Retrieving recent sessions`, { limit });

  try {
    const database = await initDb();

    // Optimize database queries by breaking down into smaller operations
    // This avoids the complex multi-table join that was causing timeouts

    // Step 1: Get basic session information first (lightweight query)
    const sessions = await monitorDbOperation('getRecentSessionsBasic', undefined, async () => {
      return database.all(
        `SELECT id, timestamp, sessionStart
         FROM sessions
         ORDER BY timestamp DESC
         LIMIT ?`,
        limit,
      );
    });

    // If no sessions found, return empty array early
    if (sessions.length === 0) {
      metricsServiceLogger.info('No sessions found');
      return [];
    }

    // Extract session IDs for subsequent queries
    const sessionIds = sessions.map((s) => s.id);
    const placeholders = sessionIds.map(() => '?').join(',');

    // Step 2: Get counts in separate queries (much faster than complex joins)
    // These queries will run in parallel
    const [engagementCounts, perfLogCounts, debugCounts, sessionMetricsData] = await Promise.all([
      monitorDbOperation('getSessionEngagementCounts', undefined, async () => {
        return database.all(
          `SELECT sessionId, COUNT(*) as count
           FROM engagement
           WHERE sessionId IN (${placeholders})
           GROUP BY sessionId`,
          ...sessionIds,
        );
      }),

      monitorDbOperation('getSessionPerfLogCounts', undefined, async () => {
        return database.all(
          `SELECT sessionId, COUNT(*) as count
           FROM perfLog
           WHERE sessionId IN (${placeholders})
           GROUP BY sessionId`,
          ...sessionIds,
        );
      }),

      monitorDbOperation('getSessionDebugCounts', undefined, async () => {
        return database.all(
          `SELECT sessionId, COUNT(*) as count
           FROM debug
           WHERE sessionId IN (${placeholders})
           GROUP BY sessionId`,
          ...sessionIds,
        );
      }),

      monitorDbOperation('getSessionMetricsData', undefined, async () => {
        return database.all(
          `SELECT sessionId, data
           FROM sessionMetrics
           WHERE sessionId IN (${placeholders})`,
          ...sessionIds,
        );
      }),
    ]);

    // Step 3: Map the counts to the sessions
    const countMap = {
      engagement: engagementCounts.reduce((map, row) => {
        map[row.sessionId] = row.count;
        return map;
      }, {}),

      perfLog: perfLogCounts.reduce((map, row) => {
        map[row.sessionId] = row.count;
        return map;
      }, {}),

      debug: debugCounts.reduce((map, row) => {
        map[row.sessionId] = row.count;
        return map;
      }, {}),
    };

    // Step 4: Map the session metrics data
    const metricsMap = sessionMetricsData.reduce((map, row) => {
      map[row.sessionId] = row.data;
      return map;
    }, {});

    // Step 5: Combine all data
    const enrichedSessions = sessions.map((session) => ({
      id: session.id,
      timestamp: session.timestamp,
      sessionStart: session.sessionStart,
      engagementCount: countMap.engagement[session.id] || 0,
      perfLogCount: countMap.perfLog[session.id] || 0,
      debugCount: countMap.debug[session.id] || 0,
      sessionMetrics: metricsMap[session.id] || null,
    }));

    const queryTime = Date.now() - queryStart;

    // Only log warning if query was slow
    if (queryTime > 5000) {
      // 5 seconds threshold
      metricsServiceLogger.warn(`Slow sessions query`, {
        count: sessions.length,
        queryTime: `${queryTime}ms`,
        firstSession: sessions[0]?.id || 'none',
      });
    } else {
      metricsServiceLogger.success(`Retrieved recent sessions`, {
        count: sessions.length,
        queryTime: `${queryTime}ms`,
        firstSession: sessions[0]?.id || 'none',
      });
    }

    return enrichedSessions;
  } catch (err) {
    const error = err as Error;
    metricsServiceLogger.error(`Failed to retrieve recent sessions`, {
      error: error.message,
      stack: error.stack,
      queryTime: `${Date.now() - queryStart}ms`,
    });

    throw error;
  }
}

/**
 * Gets key performance statistics for a specific session
 *
 * @param sessionId - ID of the session to analyze
 * @returns Performance statistics for the session
 */
type ActionStats = {
  count: number;
  avg: string;
  min: string;
  max: string;
  p95: string;
};

type SessionPerformanceStats = {
  sessionId: string;
  totalPerfLogs: number;
  distinctActions: number;
  actionStats: Record<string, ActionStats>;
};

export async function getSessionPerformanceStats(
  sessionId: string,
): Promise<SessionPerformanceStats> {
  const queryStart = Date.now();

  metricsServiceLogger.info(`Retrieving performance stats for session`, { sessionId });

  try {
    const database = await initDb();

    // Get all performance logs for the session
    const perfLogs = await monitorDbOperation('getSessionPerfLogs', sessionId, async () => {
      return database.all(
        `SELECT * FROM perfLog WHERE sessionId = ? ORDER BY timestamp ASC`,
        sessionId,
      );
    });

    // Calculate performance metrics
    const actions = perfLogs.reduce(
      (acc, log) => {
        const action = log.action;
        if (!acc[action]) {
          acc[action] = [];
        }
        acc[action].push(log.perfNow);
        return acc;
      },
      {} as Record<string, number[]>,
    );

    // Calculate statistics for each action
    const stats: Record<string, ActionStats> = {};

    for (const [action, times] of Object.entries(actions)) {
      const timesArray = times as number[];
      if (timesArray.length > 0) {
        const avg = timesArray.reduce((sum, time) => sum + time, 0) / timesArray.length;
        const min = Math.min(...timesArray);
        const max = Math.max(...timesArray);
        stats[action] = {
          count: timesArray.length,
          avg: avg.toFixed(2),
          min: min.toFixed(2),
          max: max.toFixed(2),
          p95: calculatePercentile(timesArray, 95).toFixed(2),
        };
      }
    }

    const sessionStats = {
      sessionId,
      totalPerfLogs: perfLogs.length,
      distinctActions: Object.keys(actions).length,
      actionStats: stats,
    };

    const queryTime = Date.now() - queryStart;

    metricsServiceLogger.success(`Retrieved performance stats for session`, {
      sessionId,
      queryTime: `${queryTime}ms`,
      actionCount: Object.keys(actions).length,
      totalEntries: perfLogs.length,
    });

    return sessionStats;
  } catch (err) {
    const error = err as Error;
    metricsServiceLogger.error(`Failed to retrieve performance stats for session`, {
      sessionId,
      error: error.message,
      stack: error.stack,
      queryTime: `${Date.now() - queryStart}ms`,
    });

    throw error;
  }
}

/**
 * Helper function to calculate percentiles
 */
function calculatePercentile(values: number[], percentile: number): number {
  if (values.length === 0) return 0;

  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.ceil((percentile / 100) * sorted.length) - 1;
  return sorted[index];
}
