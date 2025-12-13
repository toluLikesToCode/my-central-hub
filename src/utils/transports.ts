import winston from 'winston';
import { config } from '../config/server.config';
import path from 'path';

export const defaultFileTransport = new winston.transports.File({
  filename: path.join(config.logging.logDir, 'app.log'),
  format: winston.format.combine(winston.format.timestamp(), winston.format.json()),
});

export const defaultConsoleTransport = new winston.transports.Console({
  format: winston.format.combine(winston.format.timestamp(), winston.format.prettyPrint()),
});

export const defaultTransports = [defaultFileTransport, defaultConsoleTransport];
