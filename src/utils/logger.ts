// src/utils/logger.ts

type LogLevel = 'info' | 'warn' | 'error' | 'debug';

class Logger {
  private getTimestamp(): string {
    return new Date().toISOString();
  }

  private formatMessage(level: LogLevel, message: string): string {
    return `[${this.getTimestamp()}] [${level.toUpperCase()}]: ${message}`;
  }

  info(message: string): void {
    console.log(this.formatMessage('info', message));
  }

  warn(message: string): void {
    console.warn(this.formatMessage('warn', message));
  }

  error(message: string): void {
    console.error(this.formatMessage('error', message));
  }

  debug(message: string): void {
    if (process.env.NODE_ENV === 'development') {
      console.debug(this.formatMessage('debug', message));
    }
  }
}

// Export a singleton instance
export const logger = new Logger();
