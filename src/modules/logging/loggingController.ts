/**
 * Logging Controller Module
 *
 * Handles HTTP requests for log ingestion from remote clients.
 *
 * @module modules/logging/loggingController
 */
import { Socket } from 'net';
import { IncomingRequest } from '../../entities/http';
import { sendWithContext } from '../../entities/sendResponse';
import { storeLogEntry, LogEntry, initPromise } from './loggingService';
import { config } from '../../config/server.config';
import { getHeader } from '../../utils/httpHelpers';
import { Logger, FileTransport, ConsoleTransport, PrettyFormatter } from '../../utils/logger';
import path from 'path';
import os from 'os';

// Create specialized logger for the logging controller
const loggingControllerLogger = new Logger({
  metadata: {
    module: 'logging',
    component: 'logging-controller',
    version: '1.0.0',
    hostname: os.hostname(),
  },
  transports: [
    new ConsoleTransport({
      formatter: new PrettyFormatter({
        useColors: true,
        showTimestamp: true,
      }),
      level: 'info',
    }),
    new FileTransport({
      filename: path.join(config.logging.logDir, 'logging-controller.log'),
      formatter: new PrettyFormatter({
        useColors: false,
        showTimestamp: true,
      }),
      level: 'debug',
    }),
  ],
});

// Wait for the database to initialize before handling requests
export const __loggingDbInitPromise = initPromise;

/**
 * API tokens for authenticating log submissions
 * In a production environment, these should be loaded from a secure source
 */
const validTokens = [
  config.adminKey, // Reuse the admin key for now
  // Add additional tokens as needed
];

/**
 * Validates the authorization header
 *
 * @param authHeader The authorization header from the request
 * @returns true if the authorization is valid, false otherwise
 */
function isValidAuth(authHeader: string | undefined): boolean {
  if (!authHeader) {
    return false;
  }

  // Check for Bearer token
  if (authHeader.startsWith('Bearer ')) {
    const token = authHeader.substring(7);
    return validTokens.includes(token);
  }

  // Check for API key
  if (authHeader.startsWith('ApiKey ')) {
    const apiKey = authHeader.substring(7);
    return validTokens.includes(apiKey);
  }

  return false;
}

/**
 * Handles incoming log batches from remote clients
 *
 * @param req The incoming HTTP request
 * @param sock The socket connection
 */
export async function ingestLogs(req: IncomingRequest, sock: Socket): Promise<void> {
  await __loggingDbInitPromise;

  try {
    // 1. Authenticate if header is present
    const authHeader = getHeader(req, 'authorization');
    if (authHeader && !isValidAuth(authHeader)) {
      loggingControllerLogger.warn('Unauthorized log ingestion attempt', {
        ip: sock.remoteAddress,
        userAgent: getHeader(req, 'user-agent'),
      });

      sendWithContext(
        req,
        sock,
        401,
        {
          'Content-Type': 'application/json',
        },
        JSON.stringify({ error: 'Unauthorized' }),
      );
      return;
    }

    // 2. Ensure we have a request body
    if (!req.body) {
      sendWithContext(
        req,
        sock,
        400,
        {
          'Content-Type': 'application/json',
        },
        JSON.stringify({ error: 'Missing request body' }),
      );
      return;
    }

    // 3. Parse and validate the request body
    let logEntries: LogEntry[];
    try {
      const bodyContent = req.body.toString();
      const parsed = JSON.parse(bodyContent);

      if (!Array.isArray(parsed)) {
        throw new Error('Expected an array of log entries');
      }

      logEntries = parsed;
    } catch (error) {
      loggingControllerLogger.warn('Invalid log ingestion payload', {
        error: error instanceof Error ? error.message : String(error),
        ip: sock.remoteAddress,
      });

      sendWithContext(
        req,
        sock,
        400,
        {
          'Content-Type': 'application/json',
        },
        JSON.stringify({ error: 'Invalid request format. Expected a JSON array of log entries.' }),
      );
      return;
    }

    // 4. Validate and store each log entry
    const invalidEntries: number[] = [];
    const storedIds: number[] = [];

    for (let i = 0; i < logEntries.length; i++) {
      const entry = logEntries[i];

      // Basic validation of required fields
      if (
        typeof entry.level !== 'string' ||
        typeof entry.message === 'undefined' ||
        typeof entry.timestamp !== 'string'
      ) {
        invalidEntries.push(i);
        continue;
      }

      try {
        // Store the log entry with client information
        const id = await storeLogEntry(
          entry,
          sock.remoteAddress || undefined,
          getHeader(req, 'user-agent'),
        );
        storedIds.push(id);
      } catch (error) {
        loggingControllerLogger.error('Error storing log entry', {
          error:
            error instanceof Error ? { message: error.message, stack: error.stack } : String(error),
          entryIndex: i,
        });
        invalidEntries.push(i);
      }
    }

    // 5. Send appropriate response
    if (invalidEntries.length > 0 && invalidEntries.length === logEntries.length) {
      // All entries were invalid
      sendWithContext(
        req,
        sock,
        400,
        {
          'Content-Type': 'application/json',
        },
        JSON.stringify({ error: 'All log entries were invalid', invalidEntries }),
      );
    } else if (invalidEntries.length > 0) {
      // Some entries were invalid but some were stored
      sendWithContext(
        req,
        sock,
        207,
        {
          'Content-Type': 'application/json',
        },
        JSON.stringify({
          status: 'partial',
          stored: storedIds.length,
          invalid: invalidEntries.length,
          invalidEntries,
        }),
      );
    } else {
      // All entries were successfully stored
      loggingControllerLogger.info('Successfully ingested log entries', {
        count: logEntries.length,
        ip: sock.remoteAddress,
      });

      sendWithContext(
        req,
        sock,
        200,
        {
          'Content-Type': 'application/json',
        },
        JSON.stringify({ status: 'ok', stored: storedIds.length }),
      );
    }
  } catch (error) {
    // Handle unexpected errors
    loggingControllerLogger.error('Unexpected error in log ingestion', {
      error:
        error instanceof Error ? { message: error.message, stack: error.stack } : String(error),
      ip: sock.remoteAddress,
    });

    sendWithContext(
      req,
      sock,
      500,
      {
        'Content-Type': 'application/json',
      },
      JSON.stringify({ error: 'Internal server error while processing log entries' }),
    );
  }
}
