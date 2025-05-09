/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

import { createServer, Socket } from 'net';
import { HttpServer } from '../../src/core/server';
import { sendResponse } from '../../src/entities/sendResponse';
import logger from '../../src/utils/logger';
import { config } from '../../src/config/server.config';
import * as RouterModule from '../../src/core/router';
import { HttpRequestParser } from '../../src/core/httpParser';
import { IncomingRequest } from '../../src/entities/http';
import { on } from 'events';

jest.mock('net');
jest.mock('../../src/utils/logger');
jest.mock('../../src/entities/sendResponse');
jest.mock('../../src/modules/file-hosting/FileStatsInitializer', () => ({
  initializeFileStats: jest.fn().mockResolvedValue(undefined),
}));

jest.mock('../../src/config/server.config', () => ({
  config: {
    // Keep existing mocks
    headerTimeoutMs: 100,
    bodyTimeoutMs: 150,

    // --- Add missing properties needed by imported modules ---
    // Add the 'features' object with necessary flags (can use default values)
    features: {
      metrics: true, // Add defaults for all features
      fileHosting: true,
      fileStreaming: true, // <- Specifically needed here
      embeddingService: true,
    },
    // Add 'mediaDir' as it's also used in stream.routes.ts
    mediaDir: '/mock/media/dir', // Provide a mock path
    // Add 'logging' object structure if needed by other transitively imported files
    logging: {
      logDir: '/mock/logs', // Provide a mock path
    },
    // Add other properties used during module initialization if necessary
    // ...
    embedding: {
      // Python process settings
      maxRetries: 3,
      retryDelayMs: 1000,
      timeoutMs: 30000,
      // Embedding service settings
      serviceUrl: 'http://192.168.1.107:3456',
      pythonExecutable: 'python3',
      pythonScriptPath: '/mock/python/embedding_service_helper.py', // Path relative to project root
      pythonLogPath: '/mock/python/logs/', // Optional: Path for python script's own log, defaults to alongside script if not set
      // Model/Processing Args passed to Python script
      modelArgs: [
        '--enable_augmentation',
        '--log',
        '--debug',
        '-n',
        '30',
        // Example: '--model', 'openai/clip-vit-base-patch32' is now expected to be set here
      ],
      defaultModel: 'openai/clip-vit-base-patch32', // Default model if not in args
      defaultNumFrames: 15,
      enableAugmentation: false, // Default augmentation flag for python script
      // Service behavior
      inactivityTimeoutMs: 10 * 60 * 1000,
      scriptTimeoutMs: 30 * 60 * 1000,
      debug: true,
      log: true,
      inputDir: '/mock/embedding/dir', // Mock path for inputDir
    },
  },
}));

describe('HttpServer', () => {
  let server: HttpServer;
  let mockSocket: Socket;
  let mockNetServer: any;
  let parserInstance: any;
  let realRouter: any;

  beforeEach(() => {
    jest.clearAllMocks();

    mockSocket = {
      on: jest.fn(),
      once: jest.fn(),
      write: jest.fn(),
      end: jest.fn(),
      destroy: jest.fn(),
      setMaxListeners: jest.fn(),
      getMaxListeners: jest.fn().mockReturnValue(10),
      remoteAddress: '127.0.0.1',
    } as unknown as Socket;

    parserInstance = new HttpRequestParser();

    mockNetServer = {
      on: jest.fn(),
      listen: jest.fn(),
      close: jest.fn((cb: () => void) => cb()),
      once: jest.fn((event: string, cb: () => void) => {
        if (event === 'listening') {
          cb();
        }
      }),
    };
    (createServer as jest.Mock).mockReturnValue(mockNetServer);

    realRouter = RouterModule.createRouter();
    jest.spyOn(realRouter, 'handle').mockImplementation(jest.fn());
    server = new HttpServer(3000, realRouter);
  });

  const getConnectionHandler = () => {
    const connCall = mockNetServer.on.mock.calls.find(
      ([evt]: [string, any]) => evt === 'connection',
    );
    return connCall ? connCall[1] : null;
  };

  test('should listen on provided port', async () => {
    await server.start();
    // Accept either a string matching an IP address and a port number
    // or just a port number
    expect(mockNetServer.listen).toHaveBeenCalled();
    const listenCall = mockNetServer.listen.mock.calls[0];
    expect(listenCall[0]).toBe(3000);
    if (listenCall.length === 2) {
      expect(typeof listenCall[0]).toBe('number');
      expect(typeof listenCall[1]).toBe('string');
      expect(listenCall[1]).toMatch(/\d+\.\d+\.\d+\.\d+/);
    } else {
      // Only port
      expect(listenCall.length).toBe(1);
      expect(typeof listenCall[0]).toBe('number');
    }
  });

  test('should destroy sockets on stop()', async () => {
    const destroySpy = jest.spyOn(mockSocket, 'destroy');
    (server as any).connections.add(mockSocket);

    await server.stop();

    expect(destroySpy).toHaveBeenCalled();
    expect(mockNetServer.close).toHaveBeenCalled();
  });

  test('should handle incoming connection and process data', async () => {
    let dataHandler;
    mockSocket.on = jest.fn().mockImplementation((event, cb) => {
      if (event === 'data') dataHandler = cb;
    });
    mockSocket.once = jest.fn();

    const connHandler = getConnectionHandler();
    expect(connHandler).not.toBeNull();
    connHandler(mockSocket);

    const handleSpy = jest.spyOn(realRouter, 'handle');

    dataHandler(Buffer.from('GET / HTTP/1.1\r\nHost: localhost\r\n\r\n'));
    expect(handleSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        method: 'GET',
        path: '/',
        invalid: false,
      }),
      mockSocket,
    );
  });

  test('should destroy socket on header timeout', () => {
    let timeoutFn;
    global.setTimeout = jest.fn((fn) => {
      timeoutFn = fn;
      return 123;
    }) as any;
    global.clearTimeout = jest.fn();

    mockSocket.once = jest.fn();
    mockSocket.on = jest.fn();

    const connHandler = getConnectionHandler();
    expect(connHandler).not.toBeNull();
    connHandler(mockSocket);

    expect(setTimeout).toHaveBeenCalled();
    expect(typeof timeoutFn).toBe('function');

    timeoutFn(); // simulate timeout trigger
    expect(mockSocket.destroy).toHaveBeenCalled();
  });

  test('should destroy socket on body timeout for POST request', async () => {
    let dataHandler;
    mockSocket.on = jest.fn().mockImplementation((event, cb) => {
      if (event === 'data') dataHandler = cb;
    });
    mockSocket.once = jest.fn();

    const connHandler = getConnectionHandler();
    expect(connHandler).not.toBeNull();
    connHandler(mockSocket);

    let bodyTimeoutFn;
    global.setTimeout = jest.fn((fn) => {
      if (!bodyTimeoutFn) bodyTimeoutFn = fn;
      return 123;
    }) as any;
    global.clearTimeout = jest.fn();

    dataHandler(Buffer.from('POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\n'));
    jest.spyOn(realRouter, 'handle').mockImplementation(async () => {
      /* Does nothing */
    });

    expect(typeof bodyTimeoutFn).toBe('function');
    bodyTimeoutFn(); // simulate body timeout
    expect(mockSocket.destroy).toHaveBeenCalled();
  });

  test('should handle multiple pipelined requests in sequence', async () => {
    let dataHandler;
    mockSocket.on = jest.fn().mockImplementation((event, cb) => {
      if (event === 'data') dataHandler = cb;
    });
    mockSocket.once = jest.fn();

    const connHandler = getConnectionHandler();
    expect(connHandler).not.toBeNull();
    connHandler(mockSocket);

    const handleSpy = jest.spyOn(realRouter, 'handle');

    const requestChunk = Buffer.from(
      'GET /first HTTP/1.1\r\nHost: a\r\n\r\nGET /second HTTP/1.1\r\nHost: b\r\n\r\n',
    );
    await dataHandler(requestChunk);
    expect(handleSpy).toHaveBeenCalledTimes(2);
    expect(handleSpy).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ path: '/first' }),
      mockSocket,
    );
    expect(handleSpy).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ path: '/second' }),
      mockSocket,
    );
  });

  test('should handle socket error event', () => {
    let errorHandler: (err: Error) => void = () => {};
    mockSocket.on = jest.fn().mockImplementation((event, cb) => {
      if (event === 'error') errorHandler = cb;
    });
    mockSocket.once = jest.fn();
    const connHandler = getConnectionHandler();
    expect(connHandler).not.toBeNull();
    connHandler(mockSocket);

    const testError = new Error('Test socket error');
    (testError as NodeJS.ErrnoException).code = 'ECONNRESET';
    errorHandler(testError);

    expect(logger.error).toHaveBeenCalledWith(
      'Socket error:',
      expect.objectContaining({ error: 'Test socket error' }),
    );
    expect(mockSocket.end).not.toHaveBeenCalled();
    expect(mockSocket.destroy).not.toHaveBeenCalled();
  });

  test('should close socket if Connection: close header is present', async () => {
    let dataHandler;
    mockSocket.on = jest.fn().mockImplementation((event, cb) => {
      if (event === 'data') dataHandler = cb;
    });
    mockSocket.once = jest.fn();
    const connHandler = getConnectionHandler();
    expect(connHandler).not.toBeNull();
    connHandler(mockSocket);

    jest
      .spyOn(realRouter, 'handle')
      .mockImplementation(async (req: IncomingRequest, sock: Socket) => {});

    jest.useFakeTimers();

    await dataHandler(
      Buffer.from('GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n'),
    );

    jest.advanceTimersByTime(50);

    expect(realRouter.handle).toHaveBeenCalled();
    expect(mockSocket.end).toHaveBeenCalled();

    jest.useRealTimers();
  });
});
