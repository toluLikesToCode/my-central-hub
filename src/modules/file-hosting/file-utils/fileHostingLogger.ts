// src/modules/file-hosting/logger.ts
import path from 'path';
import { Logger, ConsoleTransport, FileTransport, PrettyFormatter } from '../../../utils/logger';
import { config } from '../../../config/server.config';

const FileHostingLogger = new Logger({
  metadata: {
    module: 'file-hosting',
  },
  level: 'silly',
  transports: [
    new ConsoleTransport({
      formatter: new PrettyFormatter({
        useColors: true,
        useBoxes: true,
        showTimestamp: true,
        maxDepth: 3,
      }),
      level: 'info',
    }),
    new FileTransport({
      filename: path.join(
        config.logging?.logDir ?? path.join(process.cwd(), 'logs'),
        'file-hosting-module.log',
      ),
      formatter: new PrettyFormatter({
        useColors: false,
        useBoxes: false,
        showTimestamp: true,
        indent: 2,
        maxDepth: 4,
        arrayLengthLimit: 10,
        objectKeysLimit: 8,
        stringLengthLimit: 200,
      }),
      level: 'silly',
    }),
  ],
});

function getLoggerInstance(metadata: Record<string, string>) {
  return FileHostingLogger.child({
    metadata,
  });
}

export default getLoggerInstance;
