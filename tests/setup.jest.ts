jest.mock('winston', () => {
  const mLogger = {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
    add: jest.fn(),
  };
  return {
    createLogger: jest.fn(() => mLogger),
    transports: {
      Console: jest.fn(),
      File: jest.fn(),
    },
    format: {
      combine: jest.fn(),
      timestamp: jest.fn(),
      prettyPrint: jest.fn(),
    },
  };
});
