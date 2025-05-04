/* eslint-disable prefer-const */
/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */
// logger.ts - Refactored modular and extensible logging utility
// Fix: Added explicit method signatures for standard levels + success to satisfy TypeScript.

import fs from 'fs';
import path from 'path';
import chalk from 'chalk'; // Dependency for PrettyFormatter
import boxen from 'boxen'; // Dependency for PrettyFormatter
import { Writable } from 'stream';
import { formatDate } from './dateFormatter'; // Import our new date formatter

// --- Configuration ---

const standardLevels = {
  error: 0,
  warn: 1,
  info: 2,
  http: 3,
  verbose: 4,
  debug: 5,
  silly: 6,
};

type LogLevel = keyof typeof standardLevels;
type CustomLevels = Record<string, number>;
// Combine standard and potential custom levels for type checking
type AllLogLevels = LogLevel | 'success' | keyof CustomLevels; // Include 'success' explicitly if used often

// --- Interfaces ---

export interface LogEntry {
  level: string;
  message: string | object;
  meta?: Record<string, any>;
  timestamp: Date;
}

export interface Formatter {
  format(entry: LogEntry): string;
}

export interface Transport {
  log(formattedMessage: string, entry: LogEntry): void;
  level?: string;
  close?(): void;
  // Add formatter property to the interface for type safety
  formatter: Formatter;
}

// --- Formatters ---

export class JsonFormatter implements Formatter {
  format(entry: LogEntry): string {
    const logObject = {
      level: entry.level,
      message: entry.message,
      timestamp: formatDate(entry.timestamp), // Use our new human-readable format
    };
    let metaBlock = '';
    if (entry.meta && Object.keys(entry.meta).length > 0) {
      metaBlock = '\nMeta: \n' + JSON.stringify(entry.meta, null, 4);
    }
    try {
      // Handle potential circular references and BigInts
      return (
        JSON.stringify(logObject, (key, value) =>
          typeof value === 'bigint' ? value.toString() : value,
        ) + metaBlock
      );
    } catch (error) {
      // Fallback for stringification errors (e.g., circular refs)
      return (
        JSON.stringify({
          level: entry.level,
          message: `[Unserializable Object: ${
            error instanceof Error ? error.message : String(error)
          }]`,
          timestamp: formatDate(entry.timestamp), // Use our new human-readable format
        }) + metaBlock
      );
    }
  }
}

/**
 * PrettyFormatter formats log entries into a human-readable, colored and optionally boxed output.
 * It supports:
 *  - Customizable options for colors, boxing, nesting depth, and truncation limits.
 *  - Semantic highlighting for file paths, error keywords, numbers, and log level names.
 *  - Flexible indenting and line-wrapping for complex objects and arrays.
 *
 * @example
 * // Basic usage with default settings:
 * const formatter = new PrettyFormatter();
 * const output = formatter.format({
 *   level: 'info',
 *   message: 'User logged in',
 *   meta: { userId: 'abc123', role: 'admin' },
 *   timestamp: new Date(),
 * });
 * console.log(output);
 *
 * @example
 * // Enable boxes and timestamps:
 * const formatter = new PrettyFormatter({ useBoxes: true, showTimestamp: true });
 * console.log(formatter.format({ level: 'error', message: 'Fatal exception', meta: {}, timestamp: new Date() }));
 *
 * @example
 * // Limit array and object key expansion:
 * const formatter = new PrettyFormatter({ arrayLengthLimit: 3, objectKeysLimit: 2, maxDepth: 2 });
 *
 * @remarks
 * The options object may include:
 *  - useColors: toggle ANSI colors (default true)
 *  - useBoxes: wrap output in a box (default false)
 *  - maxDepth: maximum recursive depth when inspecting nested data (default 3)
 *  - indent: number of spaces per indent level (default 2)
 *  - stringLengthLimit: length threshold for truncating long strings (default 150)
 *  - arrayLengthLimit: number of array items to display before truncation (default 5)
 *  - objectKeysLimit: number of object properties to display before truncation (default 5)
 *  - showTimestamp: include timestamp prefix in formatted output (default false)
 *
 * Internally, PrettyFormatter applies semantic regex-based highlights and uses chalk
 * for coloring and boxen for boxing when enabled.
 */
export class PrettyFormatter implements Formatter {
  private readonly options: {
    useColors: boolean;
    useBoxes: boolean;
    maxDepth: number;
    indent: number;
    stringLengthLimit: number;
    arrayLengthLimit: number;
    objectKeysLimit: number;
    showTimestamp: boolean;
  };

  private static LANGUAGE_COLOR_MAP: Record<string, (s: string) => string> = {
    python: chalk.magentaBright,
    typescript: chalk.cyanBright,
    javascript: chalk.yellowBright,
    shell: chalk.greenBright,
    default: chalk.whiteBright,
  };

  private static LEVEL_STYLES: Record<
    string,
    { color: (s: string) => string; icon: string; colorName: string }
  > = {
    error: { color: chalk.red, icon: '✖', colorName: 'red' },
    warn: { color: chalk.yellow, icon: '⚠', colorName: 'yellow' },
    info: { color: chalk.blueBright, icon: 'ℹ', colorName: 'blueBright' },
    success: { color: chalk.green, icon: '✔', colorName: 'green' },
    http: { color: chalk.magenta, icon: '↔', colorName: 'magenta' },
    verbose: { color: chalk.gray, icon: ' V ', colorName: 'gray' },
    debug: { color: chalk.cyan, icon: ' D ', colorName: 'cyan' },
    silly: { color: chalk.white, icon: ' S ', colorName: 'white' },
    default: { color: chalk.white, icon: ' ', colorName: 'white' },
  };

  constructor(options: Partial<PrettyFormatter['options']> = {}) {
    this.options = {
      useColors: options.useColors ?? true,
      useBoxes: options.useBoxes ?? false, // Disabled by default
      maxDepth: options.maxDepth ?? 3, // Increased default depth slightly
      indent: options.indent ?? 2,
      stringLengthLimit: options.stringLengthLimit ?? 150, // Increased limit slightly
      arrayLengthLimit: options.arrayLengthLimit ?? 5,
      objectKeysLimit: options.objectKeysLimit ?? 5,
      showTimestamp: options.showTimestamp ?? false,
    };
  }

  private detectLanguage(context?: string): string {
    if (!context) return 'default';
    if (context.endsWith('.py')) return 'python';
    if (context.endsWith('.ts')) return 'typescript';
    if (context.endsWith('.js')) return 'javascript';
    if (context.endsWith('.sh')) return 'shell';
    return 'default';
  }

  private highlightSemantics(str: string): string {
    // Avoid double-coloring numbers if useBoxes is true (boxen will color the whole box)
    if (!this.options.useColors) return str;
    try {
      const hasAnsi = new RegExp(
        // eslint-disable-next-line no-control-regex
        '(?:\\u001b\\[[0-9;]*m|\\x1b\\[[0-9;]*m|\\u001b\\[.*?m|\\x1b\\[.*?m)',
      ).test(str);
      // Highlight file paths
      str = str.replace(
        /([./\w-]+\.(?:ts|js|py|sh|json|log|txt|md|html|css))/g,
        chalk.underline.blue('$1'),
      );
      // Highlight error keywords
      str = str.replace(
        /\b(error|exception|fail(?:ed)?|traceback|stack|warn(?:ing)?|critical|fatal)\b/gi,
        chalk.bgRed.white('$1'),
      );
      // Only highlight numbers if not using boxes (to avoid double-coloring)
      if (!this.options.useBoxes && !hasAnsi) {
        str = str.replace(/\b(\d+(?:\.\d+)?)\b/g, chalk.yellow('$1'));
      }
      // Highlight common log levels in messages
      str = str.replace(/\b(INFO|DEBUG|WARN|ERROR|SUCCESS|HTTP|VERBOSE|SILLY)\b/g, (match) => {
        const style = PrettyFormatter.LEVEL_STYLES[match.toLowerCase()] || {
          color: chalk.white,
        };
        return style.color(match);
      });
    } catch (e) {
      console.error('PrettyFormatter highlighting error:', e);
    }
    return str;
  }

  private formatValue(value: any, level = 0): string {
    const pad = ' '.repeat(this.options.indent * level);
    const colorize = this.options.useColors;

    // Max depth check
    if (level > this.options.maxDepth) {
      if (Array.isArray(value)) return colorize ? chalk.gray('[Array]') : '[Array]';
      if (typeof value === 'object' && value !== null)
        return colorize ? chalk.gray('[Object]') : '[Object]';
      return colorize ? chalk.gray('...') : '...';
    }

    // Type-based formatting
    if (value === null) return colorize ? chalk.gray('null') : 'null';
    if (value === undefined) return colorize ? chalk.gray('undefined') : 'undefined';
    if (typeof value === 'string') {
      const highlighted = this.highlightSemantics(value);
      const truncated =
        highlighted.length > this.options.stringLengthLimit
          ? highlighted.slice(0, this.options.stringLengthLimit) + '...'
          : highlighted;
      // Use JSON.stringify for proper escaping, then remove surrounding quotes for readability
      const jsonStr = JSON.stringify(truncated);
      return colorize ? chalk.yellowBright(jsonStr.slice(1, -1)) : jsonStr.slice(1, -1);
    }
    if (typeof value === 'number')
      return colorize ? chalk.green(value.toString()) : value.toString();
    if (typeof value === 'boolean')
      return colorize ? chalk.magenta(value.toString()) : value.toString();
    if (typeof value === 'bigint')
      return colorize ? chalk.greenBright(value.toString() + 'n') : value.toString() + 'n';
    if (typeof value === 'function')
      return colorize ? chalk.blueBright('[Function]') : '[Function]';
    if (value instanceof Date)
      return colorize ? chalk.greenBright(formatDate(value)) : formatDate(value); // Use our new human-readable format instead of ISO
    if (value instanceof Error) {
      const stack = value.stack ? `\n${value.stack.split('\n').slice(1).join('\n')}` : ''; // Show stack excluding first line
      const formattedError = `${value.name}: ${value.message}${stack}`;
      return colorize ? chalk.redBright(this.highlightSemantics(formattedError)) : formattedError;
    }
    if (value instanceof RegExp)
      return colorize ? chalk.magentaBright(value.toString()) : value.toString();

    if (Array.isArray(value)) {
      if (value.length === 0) return '[]';
      let out = '[\n';
      const max = this.options.arrayLengthLimit;
      for (let i = 0; i < Math.min(value.length, max); i++) {
        out += pad + '  ' + this.formatValue(value[i], level + 1);
        if (i < Math.min(value.length, max) - 1 || value.length > max) out += ','; // Add comma if not last shown or if truncated
        out += '\n';
      }
      if (value.length > max) {
        const moreText = `... ${value.length - max} more item(s)`;
        out += pad + '  ' + (colorize ? chalk.gray(moreText) : moreText) + '\n';
      }
      out += pad + ']';
      return out;
    }

    if (typeof value === 'object' && value !== null) {
      // Handle potential circular references more gracefully during formatting
      try {
        const keys = Object.keys(value);
        if (keys.length === 0) return '{}';
        let out = '{\n';
        const max = this.options.objectKeysLimit;
        for (let idx = 0; idx < Math.min(keys.length, max); idx++) {
          const key = keys[idx];
          const keyStr = colorize ? chalk.cyanBright(`"${key}"`) : `"${key}"`;
          out += pad + '  ' + keyStr + ': ';
          out += this.formatValue(value[key], level + 1);
          if (idx < Math.min(keys.length, max) - 1 || keys.length > max) out += ','; // Add comma
          out += '\n';
        }
        if (keys.length > max) {
          const moreText = `... ${keys.length - max} more key(s)`;
          out += pad + '  ' + (colorize ? chalk.gray(moreText) : moreText) + '\n';
        }
        out += pad + '}';
        return out;
      } catch (e) {
        // Likely a circular reference or other object traversal issue
        return colorize
          ? chalk.redBright('[Object Formatting Error]')
          : '[Object Formatting Error]';
      }
    }

    // Fallback for other types
    return colorize ? chalk.white(String(value)) : String(value);
  }

  format(entry: LogEntry): string {
    const { level, message, meta, timestamp } = entry;
    const style = PrettyFormatter.LEVEL_STYLES[level] || PrettyFormatter.LEVEL_STYLES.default;
    let formattedMessage = typeof message === 'string' ? message : JSON.stringify(message);
    let metaBlock = '';
    if (meta && Object.keys(meta).length > 0 && meta !== message) {
      metaBlock = '\nMeta: \n' + JSON.stringify(meta, null, 4);
    }
    const timestampStr = this.options.showTimestamp ? `[${formatDate(timestamp)}] ` : ''; // Use our new human-readable format
    const levelStr = `${style.icon} ${level.toUpperCase()} `;
    let finalMessage = `${timestampStr}${levelStr}${formattedMessage}${metaBlock}`;
    if (this.options.useBoxes && this.options.useColors) {
      return boxen(finalMessage, {
        padding: 1,
        margin: { top: 0, bottom: 1, left: 0, right: 0 },
        borderStyle: 'round',
        borderColor: style.colorName,
        backgroundColor: undefined,
        title: undefined,
        titleAlignment: 'center',
      });
    } else if (this.options.useColors) {
      return style.color(finalMessage);
    } else {
      return finalMessage;
    }
  }
}

// --- Transports ---

export class ConsoleTransport implements Transport {
  public formatter: Formatter;
  public level?: string;

  constructor(options: { formatter?: Formatter; level?: string } = {}) {
    this.formatter = options.formatter ?? new PrettyFormatter({ useColors: true, useBoxes: false });
    this.level = options.level;
  }

  log(formattedMessage: string, entry: LogEntry): void {
    if (entry.level === 'error') {
      console.error(formattedMessage);
    } else if (entry.level === 'warn') {
      console.warn(formattedMessage);
    } else {
      console.log(formattedMessage);
    }
  }
}

export class FileTransport implements Transport {
  public formatter: Formatter;
  private stream: Writable;
  private filename: string;
  public level?: string;

  constructor(options: { filename: string; formatter?: Formatter; level?: string }) {
    this.filename = options.filename;
    // Default to JSON for files unless overridden
    this.formatter = options.formatter ?? new JsonFormatter();
    this.level = options.level;

    this.ensureLogDir(this.filename);
    this.stream = fs.createWriteStream(this.filename, { flags: 'a' });
    this.stream.on('error', (err) => {
      console.error(`Error writing to log file ${this.filename}:`, err);
    });
    // Handle stream closing gracefully
    this.stream.on('finish', () => {
      // console.log(`Log stream closed for ${this.filename}`);
    });
  }

  private ensureLogDir(logPath: string): void {
    const dir = path.dirname(logPath);
    try {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    } catch (err) {
      console.error(`Failed to create log directory ${dir}:`, err);
    }
  }

  log(formattedMessage: string, entry: LogEntry): void {
    this.stream.write(formattedMessage + '\n', (err) => {
      if (err) {
        console.error(`Failed to write to log stream ${this.filename}:`, err);
      }
    });
  }

  close(): void {
    // Promisify stream end for cleaner shutdown
    new Promise<void>((resolve) => {
      this.stream.end(() => resolve());
    }).catch((err) => {
      console.error(`Error closing log stream ${this.filename}:`, err);
    });
  }
}

// --- Logger Core ---

export interface LoggerOptions {
  level?: LogLevel | string;
  levels?: CustomLevels;
  transports?: Transport[];
  metadata?: Record<string, any>;
  exitOnError?: boolean;
}

/**
 * Logger provides structured, level-based logging with support for multiple transports,
 * standard and custom log levels, and scoped child loggers.
 *
 * @remarks
 * - Methods for each log level (error, warn, info, debug, success, etc.) are dynamically defined.
 * - Child loggers inherit configuration and add metadata: `const child = logger.child({module: 'auth'});`
 *
 * @example
 * ```ts
 * const log = new Logger({ level: 'debug' });
 * log.info('Server started', { port: 3000 });
 * const authLog = log.child({ module: 'auth' });
 * authLog.error('Invalid credentials', { userId });
 * ```
 */
export class Logger {
  private options: Required<Omit<LoggerOptions, 'levels'>>; // Omit levels as it's merged into this.levels
  private levels: Record<string, number>;
  private transports: Transport[];

  // --- TypeScript Fix: Explicit Method Signatures ---
  // Define methods for standard levels + 'success' so TypeScript knows they exist.
  // The actual implementation is still handled dynamically below or via the core `log` method.
  error!: (message: string | object, meta?: Record<string, any>) => void;
  warn!: (message: string | object, meta?: Record<string, any>) => void;
  info!: (message: string | object, meta?: Record<string, any>) => void;
  http!: (message: string | object, meta?: Record<string, any>) => void;
  verbose!: (message: string | object, meta?: Record<string, any>) => void;
  debug!: (message: string | object, meta?: Record<string, any>) => void;
  silly!: (message: string | object, meta?: Record<string, any>) => void;
  success!: (message: string | object, meta?: Record<string, any>) => void; // Include success if it's commonly used
  // --- End TypeScript Fix ---

  /**
   * Creates a new Logger instance with configurable levels, transports, and metadata.
   *
   * @param options - Configuration options for this logger.
   * @param options.level - Minimum level to log (default: 'info'). Only messages at this level or more severe are emitted.
   * @param options.levels - Optional map of custom level names to numeric severity. Merged with standard levels.
   * @param options.transports - Array of Transport instances (console, file, etc.). Defaults to a console transport using PrettyFormatter.
   * @param options.metadata - Global metadata injected into every log entry.
   * @param options.exitOnError - If true, process.exit() will be called after an 'error' level log.
   *
   * @remarks
   * Standard log levels (severity 0–6): error, warn, info, http, verbose, debug, silly.
   * The 'success' level is provided as an alias for info.
   * Custom levels defined via `options.levels` override or extend these defaults.
   * Transports can specify their own level threshold to filter messages independently.
   *
   * @example
   * // Default logger (console output only)
   * const log = new Logger();
   * log.info('Server started', { port: 3000 });
   *
   * @example
   * // Custom transports and metadata
   * import { FileTransport } from './utils/logger';
   * const log = new Logger({
   *   level: 'debug',
   *   levels: { critical: 0 },
   *   transports: [
   *     new ConsoleTransport({ level: 'warn' }),
   *     new FileTransport({ filename: 'app.log', level: 'debug' })
   *   ],
   *   metadata: { service: 'user-service' }
   * });
   * log.critical('Database unreachable');
   */
  constructor(options: LoggerOptions = {}) {
    // Define levels, merging standard and custom. Ensure 'success' is present if used.
    this.levels = {
      ...standardLevels,
      success: standardLevels.info,
      ...(options.levels ?? {}),
    }; // Map success to info level by default

    const defaultTransports = options.transports ?? [
      new ConsoleTransport({
        formatter: new PrettyFormatter({ useColors: true }),
      }),
      new FileTransport({
        filename: path.join(process.cwd(), process.env.LOG_DIR || 'logs', '/app.log'),
        formatter: new JsonFormatter(),
      }),
    ];

    this.options = {
      level: options.level ?? 'info',
      transports: defaultTransports,
      metadata: options.metadata ?? {},
      exitOnError: options.exitOnError ?? false,
    };

    this.transports = this.options.transports;

    // --- Dynamic Method Implementation ---
    // This part still creates the runtime methods, but TypeScript now relies on the explicit signatures above.
    Object.keys(this.levels).forEach((level) => {
      // Assign the implementation to the pre-declared properties
      (this as any)[level] = (message: string | object, meta?: Record<string, any>) => {
        this.log(level, message, meta);
      };
    });
  }

  /**
   * Emits a log entry at the given level, subject to logger and transport thresholds.
   *
   * @param level - The log level name (e.g. 'error', 'warn', 'info', 'debug', etc.).
   *   Must match a configured level; unknown levels are ignored.
   * @param message - The main log content; can be a string or an object (auto-stringified).
   * @param meta - Optional, per-entry metadata to merge with global logger metadata.
   *
   * @remarks
   * - The entry is only output if `level` is at or above the logger's own `options.level`.
   * - Each Transport may further filter by its own `transport.level` setting.
   * - The final output is formatted via the Transport's `formatter` (e.g. JSON or pretty text).
   * - Dynamic shorthand methods (`logger.info()`, `logger.error()`, etc.) delegate here.
   *
   * @example
   * // Direct use
   * logger.log('info', 'Server started', { port: 8080 });
   *
   * @example
   * // Using shorthand dynamic methods
   * logger.info('User login', { userId: 'abc123' });
   *
   * @example
   * // Logging an Error object
   * const error = new Error('Failed DB query');
   * logger.error('Database error', { error });
   */
  log(level: string, message: string | object, meta?: Record<string, any>): void {
    const levelValue = this.levels[level as string] ?? -1;
    const configuredLevelValue = this.levels[this.options.level as string] ?? this.levels.info;

    if (levelValue === -1) {
      console.warn(`Attempted to log with unknown level: "${level}"`);
      return; // Don't log unknown levels
    }

    // Check main logger level
    if (levelValue > configuredLevelValue) {
      return;
    }

    const entry: LogEntry = {
      level,
      message,
      meta: { ...this.options.metadata, ...meta },
      timestamp: new Date(),
    };

    this.transports.forEach((transport) => {
      const transportLevel = transport.level ?? this.options.level;
      const transportLevelValue =
        this.levels[transportLevel as Exclude<keyof AllLogLevels, symbol>] ?? configuredLevelValue;

      // Check transport-specific level
      if (typeof transportLevelValue === 'number' && levelValue <= transportLevelValue) {
        // Use the formatter associated with the transport
        const formattedMessage = transport.formatter.format(entry);
        try {
          transport.log(formattedMessage, entry);
        } catch (err) {
          console.error(`Error in transport ${transport.constructor.name}:`, err);
        }
      }
    });
  }

  /**
   * Creates a child logger that inherits all settings and merges additional metadata.
   *
   * @param metadata - Metadata to merge into child logger entries
   * @returns A new Logger instance with inherited configuration
   */
  child(metadata: Record<string, any>): Logger {
    // Create a new instance, inheriting options but merging metadata
    const childOptions: LoggerOptions = {
      level: this.options.level,
      levels: this.levels, // Pass the combined levels object
      transports: this.transports, // Share transports by default
      metadata: { ...this.options.metadata, ...metadata },
      exitOnError: this.options.exitOnError,
    };
    return new Logger(childOptions);
  }

  /**
   * Flushes and closes all transports, ensuring streams are ended gracefully.
   */
  close(): void {
    // Use Promise.all to wait for all streams to close
    Promise.all(
      this.transports.map((transport) => {
        if (typeof transport.close === 'function') {
          try {
            return transport.close(); // Assuming close might return a promise or be synchronous
          } catch (err) {
            console.error(`Error closing transport ${transport.constructor.name}:`, err);
            return Promise.resolve(); // Resolve even if one fails to close others
          }
        }
        return Promise.resolve();
      }),
    ).catch((err) => {
      console.error('Error during logger close:', err);
    });
  }

  // --- Placeholder for AI Integration ---
  // ... (keep the AI placeholder comment)
}

// --- Backward Compatibility & Default Export ---

const defaultLogFilePath = process.env.LOG_FILE_PATH || path.join(process.cwd(), 'logs/app.log');

const defaultLogger = new Logger({
  level: (process.env.LOG_LEVEL as LogLevel) || 'info',
  transports: [
    new ConsoleTransport({
      formatter: new PrettyFormatter({
        useColors: false,
        useBoxes: true, // Keep boxes for default console
        showTimestamp: false,
      }),
      level: (process.env.CONSOLE_LOG_LEVEL as LogLevel) || undefined,
    }),
    new FileTransport({
      filename: defaultLogFilePath,
      formatter: new PrettyFormatter({
        useColors: false,
        useBoxes: false, // No boxes for file transport
        showTimestamp: true,
        maxDepth: 4,
        stringLengthLimit: 300,
        arrayLengthLimit: 15,
        objectKeysLimit: 10,
        indent: 2,
      }), // Default file transport to JSON
      level: (process.env.FILE_LOG_LEVEL as LogLevel) || undefined,
    }),
  ],
  // levels definition already includes 'success' mapped to 'info' in the constructor
});

export default defaultLogger;
export { standardLevels };

// --- Usage Examples ---
