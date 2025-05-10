import {
  Logger,
  FileTransport,
  ConsoleTransport,
  PrettyFormatter,
  JsonFormatter,
} from './utils/logger';
import { config } from './config/server.config';
import path from 'path';

const CENTRAL_FILE_TRANSPORT_DIR = config.logging.logDir;

export class LoggerFactory {
  private static instance: LoggerFactory;
  private logger: Logger;

  private constructor() {
    this.logger = new Logger({
      transports: [
        new FileTransport({
          filename: path.join(CENTRAL_FILE_TRANSPORT_DIR, 'app.log'),
          formatter: new PrettyFormatter({
            useColors: false,
            useBoxes: false,
            showTimestamp: true,
            indent: 2,
            arrayLengthLimit: 15,
            objectKeysLimit: 10,
            maxDepth: 4,
            stringLengthLimit: 300,
          }),
        }),
        new FileTransport({
          filename: path.join(CENTRAL_FILE_TRANSPORT_DIR, 'app.json'),
          formatter: new JsonFormatter(),
        }),
        new ConsoleTransport({
          formatter: new PrettyFormatter({
            useColors: true,
            useBoxes: false,
            showTimestamp: false,
            indent: 2,
            arrayLengthLimit: 10,
            objectKeysLimit: 10,
            maxDepth: 4,
            stringLengthLimit: 300,
          }),
        }),
      ],
    });
  }

  public static getInstance(): LoggerFactory {
    if (!LoggerFactory.instance) {
      LoggerFactory.instance = new LoggerFactory();
    }
    return LoggerFactory.instance;
  }

  public getLogger(): Logger {
    return this.logger;
  }
}
