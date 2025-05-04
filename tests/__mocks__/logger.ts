/* eslint-disable @typescript-eslint/no-unused-vars */
// Manual mock for logger used in tests
const mockLogger = {
  info: jest.fn(),
  error: jest.fn(),
  warn: jest.fn(),
  debug: jest.fn(),
  child: jest.fn().mockReturnValue({
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  }),
};

// Add the logger as default export (this is what the modules are importing with `import logger from`)
module.exports = mockLogger;

// Add named exports for the Logger class and other components
class Logger {
  constructor(options = {}) {
    return mockLogger;
  }
  info = jest.fn();
  error = jest.fn();
  warn = jest.fn();
  debug = jest.fn();
  child = jest.fn().mockReturnValue({
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  });
}

class JsonFormatter {
  constructor(options = {}) {}
  format = jest.fn().mockReturnValue('{"formatted":"json"}');
}

class PrettyFormatter {
  constructor(options = {}) {}
  format = jest.fn().mockReturnValue('formatted pretty');
}

class ConsoleTransport {
  constructor(options = {}) {}
  log = jest.fn();
}

class FileTransport {
  constructor(options = {}) {}
  log = jest.fn();
}

module.exports.Logger = Logger;
module.exports.JsonFormatter = JsonFormatter;
module.exports.PrettyFormatter = PrettyFormatter;
module.exports.ConsoleTransport = ConsoleTransport;
module.exports.FileTransport = FileTransport;
