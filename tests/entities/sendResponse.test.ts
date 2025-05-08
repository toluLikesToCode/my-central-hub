/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

// tests/entities/sendResponse.test.ts
import { sendResponse, beginChunkedResponse } from '../../src/entities/sendResponse';
import logger from '../../src/utils/logger';
import { Writable, Readable } from 'stream';

jest.mock('../../src/utils/logger', () => ({
  debug: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
}));

type MockedSocket = Writable & {
  destroyed: boolean;
  end: jest.Mock<any, any>;
  destroy: jest.Mock<any, any>;
};

describe('sendResponse', () => {
  let socket: MockedSocket;

  beforeEach(() => {
    socket = new Writable({
      write(_chunk, _encoding, callback) {
        callback();
      },
    }) as MockedSocket;

    socket.write = jest.fn((_chunk, _encodingOrCallback, callbackOrUndefined) => {
      if (typeof _encodingOrCallback === 'function') {
        _encodingOrCallback();
      } else if (typeof callbackOrUndefined === 'function') {
        callbackOrUndefined();
      }
      return true;
    });
    socket.end = jest.fn((cb?: () => void) => {
      if (cb) cb();
      return socket;
    }) as unknown as jest.Mock<any, any>;
    socket.destroy = jest.fn();
    socket.destroyed = false;
    jest.clearAllMocks();
  });

  test('writes headers only if no body', () => {
    sendResponse(socket, 200, { 'Content-Type': 'text/plain' });
    expect(socket.write).toHaveBeenCalledWith(
      expect.stringContaining('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n'),
    );
    expect(socket.write).toHaveBeenCalledTimes(1);
  });

  test('writes headers and string body', () => {
    sendResponse(socket, 200, { 'Content-Type': 'text/plain' }, 'Hello');
    expect(socket.write).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining(
        'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\r\n',
      ),
    );
    expect(socket.write).toHaveBeenNthCalledWith(2, 'Hello', expect.any(Function));
  });

  test('pipes readable body to socket', () => {
    const readable = new Readable({ read() {} });
    readable.push('streamed');
    readable.push(null);
    const pipeSpy = jest.spyOn(readable, 'pipe');
    sendResponse(socket, 200, {}, readable);
    expect(pipeSpy).toHaveBeenCalledWith(socket, { end: false });
    readable.destroy();
  });

  test('attaches error handler to stream body', () => {
    const readable = new Readable({ read() {} });
    const onSpy = jest.spyOn(readable, 'on');
    sendResponse(socket, 200, {}, readable);
    expect(onSpy).toHaveBeenCalledWith('error', expect.any(Function));
    readable.destroy();
  });

  test('gracefully handles unknown status code', () => {
    sendResponse(socket, 499, { 'Content-Type': 'text/plain' }, 'Weird code');
    expect(socket.write).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining(
        'HTTP/1.1 499 Status\r\nContent-Type: text/plain\r\nContent-Length: 10\r\n\r\n',
      ),
    );
    expect(socket.write).toHaveBeenNthCalledWith(2, 'Weird code', expect.any(Function));
  });

  test('returns early if socket is destroyed', () => {
    socket.destroyed = true;
    sendResponse(socket, 200, { 'Content-Type': 'text/plain' }, 'Should not write');
    expect(socket.write).not.toHaveBeenCalled();
    expect(logger.debug).toHaveBeenCalledWith(
      '[sendResponse] Attempted to write to destroyed socket',
      { status: 200 },
    );
  });

  test('writes binary buffer correctly', () => {
    const binaryData = Buffer.from([0x01, 0x02, 0x03, 0x04]);
    sendResponse(socket, 200, { 'Content-Type': 'application/octet-stream' }, binaryData);
    expect(socket.write).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining(
        'HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: 4\r\n\r\n',
      ),
    );
    expect(socket.write).toHaveBeenNthCalledWith(2, binaryData, expect.any(Function));
    expect(logger.debug).toHaveBeenCalledWith(
      '[sendResponse] Writing object body',
      expect.objectContaining({ size: '4' }),
    );
  });

  test('pipes binary stream correctly and handles stream errors', () => {
    const readable = new Readable({ read() {} }) as jest.Mocked<Readable>;
    readable.pipe = jest.fn().mockReturnValue(socket);
    readable.on = jest.fn(readable.on.bind(readable));
    readable.once = jest.fn(readable.once.bind(readable));
    readable.pause = jest.fn();
    readable.resume = jest.fn();
    readable.destroy = jest.fn();
    sendResponse(socket, 200, { 'Content-Type': 'video/mp4' }, readable);
    expect(readable.on).toHaveBeenCalledWith('error', expect.any(Function));
    const errorHandlerCall = (readable.on as jest.Mock).mock.calls.find(
      (call) => call[0] === 'error',
    );
    expect(errorHandlerCall).toBeDefined();
    const errorHandler = errorHandlerCall[1];
    if (errorHandler) {
      const streamError = new Error('Binary stream failed');
      errorHandler(streamError);
      expect(logger.error).toHaveBeenCalledWith('[sendResponse] Stream body error caught', {
        error: 'Binary stream failed',
        status: 200,
        socketDestroyed: false,
      });
      expect(socket.destroy).toHaveBeenCalledWith(streamError);
    }
    if (!readable.destroyed) readable.destroy();
  });

  test('closes connection when "Connection: close" is set for buffer body and write callback executes', () => {
    sendResponse(socket, 200, { Connection: 'close', 'Content-Type': 'text/plain' }, 'Close Me');
    expect(socket.end).toHaveBeenCalledTimes(1);
  });

  test('closes connection when "Connection: close" is set for stream body', (done) => {
    const readable = new Readable({ read() {} }) as jest.Mocked<Readable>;
    readable.push('stream data');
    readable.push(null);
    readable.pipe = jest.fn((dest, opts) => {
      if (dest === socket && opts && opts.end) {
        process.nextTick(() => {
          if (!socket.destroyed) {
            socket.end();
          }
        });
      }
      return dest;
    }) as any;
    readable.on = jest.fn(readable.on.bind(readable));
    readable.once = jest.fn(readable.once.bind(readable));
    readable.destroy = jest.fn();
    socket.end.mockImplementation(() => {
      done();
      return socket;
    });
    sendResponse(socket, 200, { Connection: 'close', 'Content-Type': 'text/plain' }, readable);
    expect(readable.pipe).toHaveBeenCalledWith(socket, { end: true });
  });

  test('does not close connection if "Connection: close" is absent for string body', () => {
    sendResponse(socket, 200, { 'Content-Type': 'text/plain' }, 'Keep Alive');
    expect(socket.end).not.toHaveBeenCalled();
  });

  test('handles errors during socket.write for headers', () => {
    const writeError = new Error('Socket header write failed');
    socket.write.mockImplementationOnce(() => {
      throw writeError;
    });
    socket.on('error', () => {});
    expect(() => {
      sendResponse(socket, 200, { 'Content-Type': 'text/plain' }, 'Data');
    }).not.toThrow();
    expect(logger.error).toHaveBeenCalledWith(
      '[sendResponse] General error during response sending',
      expect.objectContaining({ error: 'Socket header write failed' }),
    );
    expect(socket.destroy).toHaveBeenCalledWith(writeError);
  });

  test('handles errors during socket.write for body', () => {
    const writeError = new Error('Socket body write failed');
    socket.write
      .mockImplementationOnce((_data, _encodingOrCallback, cb) => {
        if (typeof _encodingOrCallback === 'function') _encodingOrCallback();
        else if (typeof cb === 'function') cb();
        return true;
      })
      .mockImplementationOnce(() => {
        throw writeError;
      });
    socket.on('error', () => {});
    expect(() => {
      sendResponse(socket, 200, { 'Content-Type': 'text/plain' }, 'Data');
    }).not.toThrow();
    expect(logger.error).toHaveBeenCalledWith(
      '[sendResponse] General error during response sending',
      expect.objectContaining({ error: 'Socket body write failed' }),
    );
    expect(socket.destroy).toHaveBeenCalledWith(writeError);
  });
});

// --- Start of beginChunkedResponse Tests ---
describe('beginChunkedResponse', () => {
  let socket: MockedSocket;

  beforeEach(() => {
    socket = new Writable({
      write(_chunk, _encoding, callback) {
        callback();
      },
    }) as MockedSocket;
    socket.write = jest.fn((_chunk, _encodingOrCallback, _callbackOrUndefined) => {
      if (typeof _encodingOrCallback === 'function') {
        _encodingOrCallback();
      } else if (typeof _callbackOrUndefined === 'function') {
        _callbackOrUndefined();
      }
      return true; // Default to returning true, implying success and data flushed
    });
    socket.end = jest.fn();
    socket.destroy = jest.fn();
    socket.destroyed = false;
    jest.clearAllMocks();
  });

  test('should write initial headers for chunked response', () => {
    beginChunkedResponse(socket, { status: 200, headers: { 'Content-Type': 'application/json' } });

    expect(socket.write).toHaveBeenCalledTimes(1);
    const writtenData = (socket.write as jest.Mock).mock.calls[0][0];

    // Corrected: Use .toContain for substring checks and .toMatch for regex
    expect(writtenData).toContain('HTTP/1.1 200 OK\r\n');
    expect(writtenData).toContain('Transfer-Encoding: chunked\r\n');
    expect(writtenData).toContain('Connection: keep-alive\r\n');
    expect(writtenData).toContain('Cache-Control: no-cache, no-store, must-revalidate\r\n');
    expect(writtenData).toContain('Content-Type: application/json\r\n');
    expect(writtenData).toMatch(/\r\n\r\n$/); // Ends with a double CRLF (end of headers)
    expect(writtenData).not.toContain('Content-Length:');
  });

  test('should include X-Request-ID and X-Accel-Buffering headers if provided', () => {
    beginChunkedResponse(socket, {
      status: 200,
      headers: {},
      requestId: 'req-123',
      noBuffering: true,
    });
    const writtenData = (socket.write as jest.Mock).mock.calls[0][0];
    // Corrected: Use .toContain for substring checks
    expect(writtenData).toContain('X-Request-ID: req-123\r\n');
    expect(writtenData).toContain('X-Accel-Buffering: no\r\n');
  });

  test('sendChunk should format and write chunk correctly for JSON data', () => {
    const { sendChunk } = beginChunkedResponse(socket, { status: 200, headers: {} });
    const data = { message: 'hello' };
    const jsonChunk = JSON.stringify(data);
    const expectedLengthHex = Buffer.byteLength(jsonChunk).toString(16);
    sendChunk(data);
    expect(socket.write).toHaveBeenNthCalledWith(2, `${expectedLengthHex}\r\n${jsonChunk}\r\n`);
  });

  test('sendChunk should format and write chunk correctly for string data', () => {
    const { sendChunk } = beginChunkedResponse(socket, { status: 200, headers: {} });
    const dataString = 'This is a string chunk';
    const expectedLengthHex = Buffer.byteLength(dataString).toString(16);
    sendChunk(dataString);
    expect(socket.write).toHaveBeenNthCalledWith(2, `${expectedLengthHex}\r\n${dataString}\r\n`);
  });

  test('endResponse should write final chunk marker and normally not end keep-alive socket', () => {
    const { endResponse } = beginChunkedResponse(socket, { status: 200, headers: {} });
    endResponse();
    expect(socket.write).toHaveBeenNthCalledWith(2, '0\r\n\r\n');
    expect(socket.end).not.toHaveBeenCalled();
  });

  test('endResponse should write final chunk marker and end socket if Connection: close is set', () => {
    const { endResponse } = beginChunkedResponse(socket, {
      status: 200,
      headers: { Connection: 'close' },
    });
    endResponse();
    expect(socket.write).toHaveBeenNthCalledWith(2, '0\r\n\r\n');
    expect(socket.end).toHaveBeenCalled();
  });

  test('should return no-op functions if socket is destroyed initially', () => {
    socket.destroyed = true;
    const { sendChunk, endResponse } = beginChunkedResponse(socket, { status: 200, headers: {} });
    expect(socket.write).not.toHaveBeenCalled();
    let chunkResult;
    expect(() => (chunkResult = sendChunk({ data: 1 }))).not.toThrow();
    expect(chunkResult).toBe(false);
    expect(() => endResponse()).not.toThrow();
    expect(socket.write).not.toHaveBeenCalled();
    expect(socket.end).not.toHaveBeenCalled();
  });

  test('sendChunk should do nothing and return false if socket becomes destroyed', () => {
    const { sendChunk } = beginChunkedResponse(socket, { status: 200, headers: {} });
    expect(socket.write).toHaveBeenCalledTimes(1);
    socket.destroyed = true;
    const result = sendChunk({ data: 1 });
    expect(result).toBe(false);
    expect(socket.write).toHaveBeenCalledTimes(1);
  });

  test('endResponse should do nothing if socket becomes destroyed after initial write', () => {
    const { endResponse } = beginChunkedResponse(socket, { status: 200, headers: {} });
    expect(socket.write).toHaveBeenCalledTimes(1);
    socket.destroyed = true;
    endResponse();
    expect(socket.write).toHaveBeenCalledTimes(1);
    expect(socket.end).not.toHaveBeenCalled();
  });

  test('should handle errors during initial header write', () => {
    const writeError = new Error('Chunked header write failed');
    (socket.write as jest.Mock).mockImplementationOnce(() => {
      throw writeError;
    });
    socket.on('error', () => {});
    expect(() => beginChunkedResponse(socket, { status: 500, headers: {} })).not.toThrow();
    expect(logger.error).toHaveBeenCalledWith(
      '[beginChunkedResponse] Error initializing chunked response',
      expect.objectContaining({ error: 'Chunked header write failed' }),
    );
    expect(socket.destroy).toHaveBeenCalledWith(writeError);
    expect(socket.end).not.toHaveBeenCalled();
  });

  test('sendChunk should handle errors during chunk write and destroy socket', () => {
    const { sendChunk } = beginChunkedResponse(socket, { status: 200, headers: {} });
    const chunkWriteError = new Error('Chunk write failed');
    (socket.write as jest.Mock).mockImplementationOnce(() => {
      throw chunkWriteError;
    });
    socket.on('error', () => {});
    const result = sendChunk({ message: 'test' });
    expect(result).toBe(false);
    expect(logger.error).toHaveBeenCalledWith(
      '[beginChunkedResponse] Error writing chunk',
      expect.objectContaining({ error: 'Chunk write failed' }),
    );
    expect(socket.destroy).toHaveBeenCalledWith(chunkWriteError);
  });

  test('endResponse should handle errors during final chunk write and destroy socket', () => {
    const { endResponse } = beginChunkedResponse(socket, { status: 200, headers: {} });
    const finalChunkWriteError = new Error('Final chunk write failed');
    (socket.write as jest.Mock).mockImplementationOnce(() => {
      throw finalChunkWriteError;
    });
    socket.on('error', () => {});
    endResponse();
    expect(logger.error).toHaveBeenCalledWith(
      '[beginChunkedResponse] Error writing final chunk',
      expect.objectContaining({ error: 'Final chunk write failed' }),
    );
    expect(socket.destroy).toHaveBeenCalledWith(finalChunkWriteError);
  });
});
