// Map to track clientSessionId header to DB sessionId and last used time
interface ClientSessionInfo {
  sessionId: number;
  lastUsed: number;
}
const clientSessionMap: Record<string, ClientSessionInfo> = {};
const CLIENT_SESSION_TTL = 1000 * 60 * 60; // 1 hour

// Periodically remove stale entries
setInterval(() => {
  const now = Date.now();
  for (const key of Object.keys(clientSessionMap)) {
    if (now - clientSessionMap[key].lastUsed > CLIENT_SESSION_TTL) {
      delete clientSessionMap[key];
      logger.info(`Cleaned up session mapping for ${key}`);
    }
  }
}, CLIENT_SESSION_TTL);

import { z } from 'zod';
import sqlite3 from 'sqlite3';
import { open, Database } from 'sqlite';
import { config } from '../../config/server.config';
import { logger } from '../../utils/logger';
import fs from 'fs';
import path from 'path';

type PerfLogEntry = {
  timestamp: string;
  perfNow: number;
  memory?: {
    usedJSHeapSize?: number;
    totalJSHeapSize?: number;
    jsHeapSizeLimit?: number;
  };
  action?: string;
  [key: string]: unknown;
};

export function isPerfLogArray(arr: unknown): arr is PerfLogEntry[] {
  if (!Array.isArray(arr)) return false;
  return arr.every(
    (item) =>
      item &&
      typeof item === 'object' &&
      'perfNow' in item &&
      typeof (item as PerfLogEntry).perfNow === 'number' &&
      'timestamp' in item,
  );
}

// --- Runtime validation schema ---
export const metricsPayloadSchema = z.object({
  engagement: z.record(
    z.object({
      views: z.number().int(),
      lastViewedAt: z.number().int(),
      totalWatchMs: z.number().int(),
      completions: z.number().int(),
    }),
  ),
  perfLog: z.array(
    z
      .object({
        timestamp: z.string(),
        perfNow: z.number(),
        memory: z
          .object({
            usedJSHeapSize: z.number().optional(),
            totalJSHeapSize: z.number().optional(),
            jsHeapSizeLimit: z.number().optional(),
          })
          .optional(),
        action: z.string().optional(),
      })
      .passthrough(),
  ),
  lastRows: z.array(
    z.object({
      items: z.array(z.any()),
      center: z.number(),
    }),
  ),
  debug: z.array(
    z.object({
      message: z.string(),
      timestamp: z.string(),
    }),
  ),
  timestamp: z.string(),
  sessionStart: z.number().int(),
  sessionMetrics: z
    .object({
      sessionId: z.string(),
      startTime: z.number().int(),
      firstClickTime: z.number().int().nullable().optional(),
      scrollDistance: z.number(),
      itemsLoaded: z.number().int(),
      infiniteScrollLoads: z.number().int(),
      hoverThumbnails: z.number().int(),
      gridClickOpenCount: z.number().int(),
      modalsOpened: z.number().int(),
      modalTotalTime: z.number(),
      carouselNavigationCount: z.number().int(),
      modalContentCounts: z.object({
        video: z.number().int(),
        image: z.number().int(),
      }),
      videoMetrics: z.object({
        plays: z.number().int(),
        completions: z.number().int(),
        watchTime: z.number(),
        manualStarts: z.number().int(),
        autoPlays: z.number().int(),
      }),
      performanceMetrics: z.object({
        preloadDurations: z.array(z.number()),
        longTasks: z.number().int(),
        infiniteLoadTimes: z.array(z.number()),
        modalAnimationLatencies: z.array(z.number()),
      }),
    })
    .passthrough()
    .optional(),
});

export type MetricsPayload = z.infer<typeof metricsPayloadSchema>;
// For sessionMetrics type
export type SessionMetrics = MetricsPayload['sessionMetrics'];

let db: Database | null = null;

/** Lazily open & migrate the DB. */
export async function initDb(): Promise<Database> {
  if (db) return db;

  const dbDir = path.dirname(config.dbPath);
  if (!fs.existsSync(dbDir)) {
    fs.mkdirSync(dbDir, { recursive: true });
    logger.info(`Created database directory at ${dbDir}`);
  }
  // Ensure the database file exists by opening it in readwrite/create mode
  if (!fs.existsSync(config.dbPath)) {
    fs.closeSync(fs.openSync(config.dbPath, 'w'));
    logger.info(`Created new SQLite database file at ${config.dbPath}`);
  }

  db = await open({ filename: config.dbPath, driver: sqlite3.Database });
  // TODO: Use umzug for migrations instead of raw SQL
  await db.exec(`
    CREATE TABLE IF NOT EXISTS sessions (
      id             INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp      TEXT,
      sessionStart   INTEGER
    );
    CREATE TABLE IF NOT EXISTS engagement (
      id             INTEGER PRIMARY KEY AUTOINCREMENT,
      sessionId      INTEGER,
      itemId         TEXT,
      views          INTEGER,
      lastViewedAt   INTEGER,
      totalWatchMs   INTEGER,
      completions    INTEGER,
      FOREIGN KEY(sessionId) REFERENCES sessions(id)
    );
    CREATE TABLE IF NOT EXISTS perfLog (
      id             INTEGER PRIMARY KEY AUTOINCREMENT,
      sessionId      INTEGER,
      timestamp      TEXT,
      perfNow        REAL,
      usedJSHeapSize INTEGER,
      totalJSHeapSize INTEGER,
      jsHeapSizeLimit INTEGER,
      action         TEXT,
      otherData      TEXT,
      FOREIGN KEY(sessionId) REFERENCES sessions(id)
    );
    CREATE TABLE IF NOT EXISTS lastRows (
      id             INTEGER PRIMARY KEY AUTOINCREMENT,
      sessionId      INTEGER,
      items          TEXT,
      center         REAL,
      FOREIGN KEY(sessionId) REFERENCES sessions(id)
    );
    CREATE TABLE IF NOT EXISTS debug (
      id             INTEGER PRIMARY KEY AUTOINCREMENT,
      sessionId      INTEGER,
      message        TEXT,
      timestamp      TEXT,
      FOREIGN KEY(sessionId) REFERENCES sessions(id)
    );
    CREATE TABLE IF NOT EXISTS sessionMetrics (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      sessionId INTEGER,
      data TEXT,
      FOREIGN KEY(sessionId) REFERENCES sessions(id)
    );
  `);
  return db;
}

/** Persist the entire payload in multiple tables. Accepts a single object or array of objects. */
export async function saveMetrics(
  payload: PerfLogEntry[] | MetricsPayload | MetricsPayload[],
  sessionMetrics: SessionMetrics | undefined,
  clientSessionId?: string,
): Promise<void> {
  // log the incoming payload shape for debugging
  logger.info(`saveMetrics invoked with payload: ${JSON.stringify(payload).slice(0, 1000)}`);
  logger.info(`isPerfLogArray: ${isPerfLogArray(payload)}`);
  const database = await initDb();
  if (isPerfLogArray(payload)) {
    logger.info(`perf-batch: ${payload.length}`);
    // handle perfLog entries only
    let sessionId: number | undefined;
    if (clientSessionId && clientSessionMap[clientSessionId] != null) {
      sessionId = clientSessionMap[clientSessionId].sessionId;
      clientSessionMap[clientSessionId].lastUsed = Date.now();
    } else {
      const sessionRow = await database.get('SELECT id FROM sessions ORDER BY id DESC LIMIT 1');
      sessionId = sessionRow?.id;
    }
    if (!sessionId) {
      // If no session found, create a new session
      const timestampNow = new Date().toISOString();
      const sessionStartNow = Date.now();
      const res = await database.run(
        `INSERT INTO sessions (timestamp, sessionStart) VALUES (?, ?)`,
        timestampNow,
        sessionStartNow,
      );
      sessionId = res.lastID!;
      if (clientSessionId) {
        clientSessionMap[clientSessionId] = { sessionId, lastUsed: Date.now() };
      }
    }
    const pStmt = await database.prepare(`
      INSERT INTO perfLog
        (sessionId, timestamp, perfNow, usedJSHeapSize, totalJSHeapSize, jsHeapSizeLimit, action, otherData)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    for (const entry of payload) {
      await pStmt.run(
        sessionId,
        entry.timestamp,
        entry.perfNow,
        entry.memory?.usedJSHeapSize ?? null,
        entry.memory?.totalJSHeapSize ?? null,
        entry.memory?.jsHeapSizeLimit ?? null,
        entry.action ?? null,
        JSON.stringify(entry),
      );
    }
    await pStmt.finalize();
    return;
  }
  const payloads = Array.isArray(payload) ? payload : [payload];
  for (const p of payloads) {
    const {
      timestamp,
      sessionStart,
      engagement,
      perfLog,
      lastRows,
      debug,
      sessionMetrics: metricsFromPayload,
    } = p;

    // reuse existing session if same sessionStart
    let sessionId: number;
    const existing = await database.get(
      'SELECT id FROM sessions WHERE sessionStart = ?',
      sessionStart,
    );
    if (existing && existing.id != null) {
      sessionId = existing.id;
    } else {
      const res = await database.run(
        `INSERT INTO sessions (timestamp, sessionStart) VALUES (?, ?)`,
        timestamp,
        sessionStart,
      );
      if (res.lastID === undefined) {
        throw new Error('Failed to insert session and retrieve lastID');
      }
      sessionId = res.lastID;
    }
    // Map header sessionId to DB sessionId and update lastUsed
    if (clientSessionId) {
      clientSessionMap[clientSessionId] = { sessionId, lastUsed: Date.now() };
    }

    // Insert sessionMetrics, prefer the argument, fallback to payload property
    const sessionMetricsToSave = sessionMetrics ?? metricsFromPayload;
    if (sessionMetricsToSave) {
      await database.run(
        `INSERT INTO sessionMetrics (sessionId, data) VALUES (?, ?)`,
        sessionId,
        JSON.stringify(sessionMetricsToSave),
      );
    }

    // 2) engagement
    const eStmt = await database.prepare(`
      INSERT INTO engagement
        (sessionId, itemId, views, lastViewedAt, totalWatchMs, completions)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    for (const [itemId, stats] of Object.entries(engagement || {})) {
      const s = stats as {
        views?: number;
        lastViewedAt?: number;
        totalWatchMs?: number;
        completions?: number;
      };
      await eStmt.run(
        sessionId,
        itemId,
        s.views ?? null,
        s.lastViewedAt ?? null,
        s.totalWatchMs ?? null,
        s.completions ?? null,
      );
    }
    await eStmt.finalize();

    // 3) perfLog
    const pStmt = await database.prepare(`
      INSERT INTO perfLog
        (sessionId, timestamp, perfNow, usedJSHeapSize, totalJSHeapSize, jsHeapSizeLimit, action, otherData)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    for (const entry of perfLog || []) {
      await pStmt.run(
        sessionId,
        entry.timestamp,
        entry.perfNow,
        entry.memory?.usedJSHeapSize ?? null,
        entry.memory?.totalJSHeapSize ?? null,
        entry.memory?.jsHeapSizeLimit ?? null,
        entry.action ?? null,
        JSON.stringify(entry),
      );
    }
    await pStmt.finalize();

    // 4) lastRows
    const rStmt = await database.prepare(`
      INSERT INTO lastRows (sessionId, items, center) VALUES (?, ?, ?)
    `);
    for (const row of lastRows || []) {
      await rStmt.run(sessionId, JSON.stringify(row.items), row.center);
    }
    await rStmt.finalize();

    // 5) debug
    const dStmt = await database.prepare(`
      INSERT INTO debug (sessionId, message, timestamp) VALUES (?, ?, ?)
    `);
    for (const d of debug || []) {
      await dStmt.run(sessionId, d.message, d.timestamp);
    }
    await dStmt.finalize();
  }
}

// Graceful shutdown: close DB connection
process.on('SIGINT', async () => {
  if (db) await db.close();
  logger.info('SQLite connection closed (SIGINT)');
  process.exit(0);
});
process.on('SIGTERM', async () => {
  if (db) await db.close();
  logger.info('SQLite connection closed (SIGTERM)');
  process.exit(0);
});
