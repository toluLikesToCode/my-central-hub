import * as winston from 'winston';
import { config } from '../config/server.config';
import path from 'path';

const isTestEnv = process.env.NODE_ENV === 'test';

// Disable file writes in test runs to avoid noisy I/O and missing format bindings.
export const defaultFileTransport = new winston.transports.File({
  filename: path.join(config.logging.logDir, 'app.log'),
  format: winston.format.combine(winston.format.timestamp(), winston.format.json()),
  silent: isTestEnv,
});

export const defaultConsoleTransport = new winston.transports.Console({
  format: winston.format.combine(winston.format.timestamp(), winston.format.prettyPrint()),
  silent: isTestEnv, // keep console quiet in tests; jest captures explicitly where needed
});

export const defaultTransports = [defaultFileTransport, defaultConsoleTransport];
