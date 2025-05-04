import { z } from 'zod';
import sqlite3 from 'sqlite3';
import { open, Database } from 'sqlite';
import { config } from '../../../config/server.config';
import { Logger, FileTransport, ConsoleTransport, PrettyFormatter } from '../../../utils/logger';
import fs from 'fs';
import path from 'path';

const logger = new Logger({
  transports: [
    new ConsoleTransport({
      formatter: new PrettyFormatter(),
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
  if (db) return db;
  // Compute the directory that should contain the .db file:
  const dbDir = path.dirname(config.dbPath);
  if (!fs.existsSync(dbDir)) {
    fs.mkdirSync(dbDir, { recursive: true });
  }
  if (!fs.existsSync(config.dbPath)) {
    fs.closeSync(fs.openSync(config.dbPath, 'w'));
    logger.info(`Created new SQLite database file at ${config.dbPath}`);
  }
  db = await open({ filename: config.dbPath, driver: sqlite3.Database });
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
  `);
  return db;
}

// --- Utility: Accept and log all valid data, never reject whole request ---
function logInvalid(type: string, data: unknown, error: unknown) {
  logger.info(`[metrics] Invalid Entry`, [type, data, error]);
}

// --- Session Helper ---
async function ensureSessionExists(
  database: Database,
  sessionId: string,
  timestamp?: string,
  sessionStart?: number,
) {
  // Try to find session by id
  const sessionRow = await database.get('SELECT id FROM sessions WHERE id = ?', sessionId);
  if (!sessionRow) {
    await database.run(
      'INSERT INTO sessions (id, timestamp, sessionStart) VALUES (?, ?, ?)',
      sessionId,
      timestamp || new Date().toISOString(),
      sessionStart ?? null,
    );
    logger.info(`[metrics] Created minimal session for id=${sessionId}`);
  }
}

export async function saveMetrics(
  payload: unknown,
  clientSessionId?: string,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _env?: unknown,
) {
  logger.info('[metrics] saveMetrics called', {
    clientSessionId,
    payloadType: Array.isArray(payload) ? 'PerfLogBatch' : typeof payload,
  });
  const database = await initDb();
  // PerfLog batch
  if (Array.isArray(payload)) {
    logger.debug('[metrics] Processing PerfLog batch', { batchSize: payload.length });
    for (const entry of payload) {
      const result = PerfLogEntrySchema.safeParse(entry);
      if (!result.success) {
        logInvalid('perfLog', entry, result.error.format());
        continue;
      }
      // Resolve sessionId
      const resolvedSessionId = result.data.sessionId || clientSessionId;
      if (!resolvedSessionId) {
        logger.warn('[metrics] No sessionId found for perfLog entry. Skipping.', entry);
        continue;
      }
      logger.debug('[metrics] Inserting PerfLog entry', { resolvedSessionId, entry });
      await ensureSessionExists(database, resolvedSessionId, result.data.timestamp);
      const { timestamp, perfNow, memory, action, batchId, uploadMode, ...rest } = result.data;
      const details = JSON.stringify({ ...rest });
      logger.debug('[metrics] Inserting PerfLog entry', result);
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
    }
    logger.info('[metrics] Finished processing PerfLog batch', { batchSize: payload.length });
    return;
  }
  // Full payload
  const result = FullPayloadSchema.safeParse(payload);
  if (!result.success) {
    logInvalid('fullPayload', payload, result.error.format());
    // Try to salvage valid subfields
    if (payload && typeof payload === 'object') {
      // Try perfLog
      const perfLogArray = Array.isArray((payload as Record<string, unknown>).perfLog)
        ? ((payload as Record<string, unknown>).perfLog as unknown[])
        : [];
      logger.debug('[metrics] Salvaging perfLog from invalid fullPayload', {
        perfLogCount: perfLogArray.length,
      });
      for (const entry of perfLogArray) {
        const perfResult = PerfLogEntrySchema.safeParse(entry);
        if (perfResult.success) {
          const { timestamp, perfNow, memory, action, batchId, uploadMode, ...rest } =
            perfResult.data;
          const details = JSON.stringify({ ...rest });
          await database.run(
            `INSERT INTO perfLog (sessionId, timestamp, perfNow, usedJSHeapSize, totalJSHeapSize, jsHeapSizeLimit, action, batchId, uploadMode, details)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
            null,
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
        } else {
          logInvalid('perfLog', entry, perfResult.error.format());
        }
      }
      // Try debug
      const debugArray = Array.isArray((payload as Record<string, unknown>).debug)
        ? ((payload as Record<string, unknown>).debug as unknown[])
        : [];
      logger.debug('[metrics] Salvaging debug logs from invalid fullPayload', {
        debugCount: debugArray.length,
      });
      for (const entry of debugArray) {
        const debugResult = DebugLogSchema.safeParse(entry);
        if (debugResult.success) {
          await database.run(
            `INSERT INTO debug (sessionId, message, timestamp) VALUES (?, ?, ?)`,
            null,
            debugResult.data.message,
            debugResult.data.timestamp,
          );
        } else {
          logInvalid('debug', entry, debugResult.error.format());
        }
      }
      // Try engagement
      const engagementObj = (payload as Record<string, unknown>).engagement;
      if (engagementObj && typeof engagementObj === 'object' && !Array.isArray(engagementObj)) {
        logger.debug('[metrics] Salvaging engagement from invalid fullPayload', {
          engagementKeys: Object.keys(engagementObj),
        });
        for (const [itemId, stats] of Object.entries(engagementObj)) {
          const statsResult = EngagementStatsSchema.safeParse(stats);
          if (statsResult.success) {
            await database.run(
              `INSERT INTO engagement (sessionId, itemId, views, lastViewedAt, totalWatchMs, completions) VALUES (?, ?, ?, ?, ?, ?)`,
              null,
              itemId,
              statsResult.data.views,
              statsResult.data.lastViewedAt,
              statsResult.data.totalWatchMs,
              statsResult.data.completions,
            );
          } else {
            logInvalid('engagement', stats, statsResult.error.format());
          }
        }
      }
    }
    return;
  }
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
  logger.info('[metrics] Inserting full payload', {
    resolvedSessionId,
    engagementCount: Object.keys(engagement).length,
    perfLogCount: perfLog.length,
    debugCount: debug.length,
  });
  await ensureSessionExists(
    database,
    resolvedSessionId || 'unknown-session',
    timestamp,
    sessionStart,
  );
  // Insert engagement
  for (const [itemId, stats] of Object.entries(engagement)) {
    logger.debug('[metrics] Inserting engagement', { resolvedSessionId, itemId, stats });
    await database.run(
      `INSERT INTO engagement (sessionId, itemId, views, lastViewedAt, totalWatchMs, completions) VALUES (?, ?, ?, ?, ?, ?)`,
      resolvedSessionId,
      itemId,
      stats.views,
      stats.lastViewedAt,
      stats.totalWatchMs,
      stats.completions,
    );
  }
  // Insert perfLog
  for (const entry of perfLog) {
    logger.debug('[metrics] Inserting perfLog', { resolvedSessionId, entry });
    const { timestamp, perfNow, memory, action, batchId, uploadMode, ...rest } = entry;
    const details = JSON.stringify({ ...rest });
    await database.run(
      `INSERT INTO perfLog (sessionId, timestamp, perfNow, usedJSHeapSize, totalJSHeapSize, jsHeapSizeLimit, action, batchId, uploadMode, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
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
  }
  // Insert debug
  for (const entry of debug) {
    logger.debug('[metrics] Inserting debug log', { resolvedSessionId, entry });
    await database.run(
      `INSERT INTO debug (sessionId, message, timestamp) VALUES (?, ?, ?)`,
      resolvedSessionId,
      entry.message,
      entry.timestamp,
    );
  }
  // Insert sessionMetrics
  logger.debug('[metrics] Inserting sessionMetrics', {
    resolvedSessionId,
    sessionMetrics: sessionMetricsFromPayload,
  });
  await database.run(
    `INSERT INTO sessionMetrics (sessionId, data) VALUES (?, ?)`,
    resolvedSessionId,
    JSON.stringify(sessionMetricsFromPayload),
  );
}

// Graceful shutdown
process.on('SIGINT', async () => {
  logger.info(`[shutdown] SIGINT received. Initiating graceful shutdown.`, {
    pid: process.pid,
    time: new Date().toISOString(),
  });
  if (db) {
    logger.debug(
      '[shutdown] Database connection detected. Attempting to close SQLite connection.',
      { hasDb: true },
    );
    try {
      await db.close();
      logger.info(`SQLite connection closed successfully (SIGINT)`, {
        event: 'SIGINT',
        closed: true,
      });
    } catch (err) {
      logger.error(`Error closing SQLite connection (SIGINT)`, { error: err, event: 'SIGINT' });
    }
  } else {
    logger.warn('No database connection found during SIGINT shutdown.', { hasDb: false });
  }
  logger.info(`Exiting process (SIGINT).`, { pid: process.pid });
  process.exit(0);
});
process.on('SIGTERM', async () => {
  logger.info(`[shutdown] SIGTERM received. Initiating graceful shutdown.`, {
    pid: process.pid,
    time: new Date().toISOString(),
  });
  if (db) {
    logger.debug(
      '[shutdown] Database connection detected. Attempting to close SQLite connection.',
      { hasDb: true },
    );
    try {
      await db.close();
      logger.info(`SQLite connection closed successfully (SIGTERM)`, {
        event: 'SIGTERM',
        closed: true,
      });
    } catch (err) {
      logger.error(`Error closing SQLite connection (SIGTERM)`, { error: err, event: 'SIGTERM' });
    }
  } else {
    logger.warn('No database connection found during SIGTERM shutdown.', { hasDb: false });
  }
  logger.info(`Exiting process (SIGTERM).`, { pid: process.pid });
  process.exit(0);
});
