/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

// tests/entities/sendResponse.test.ts
import { sendResponse } from '../../src/entities/sendResponse';
import { Socket } from 'net';
import { Readable, Writable } from 'stream';

describe('sendResponse', () => {
  let socket: jest.Mocked<Socket>;

  beforeEach(() => {
    socket = new Writable({
      write(chunk, encoding, callback) {
        callback();
      },
    }) as jest.Mocked<Socket>;

    socket.write = jest.fn();
    socket.destroy = jest.fn();
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
    expect(socket.write).toHaveBeenNthCalledWith(1, expect.stringContaining('HTTP/1.1 200 OK'));
    expect(socket.write).toHaveBeenNthCalledWith(2, 'Hello');
  });

  test('pipes readable body to socket', () => {
    const readable = new Readable();
    readable._read = () => {};
    readable.push('streamed');
    readable.push(null);

    const pipe = jest.spyOn(readable, 'pipe');
    sendResponse(socket, 200, {}, readable);

    expect(pipe).toHaveBeenCalledWith(socket, { end: false });
  });

  test('attaches error handler to stream', () => {
    const readable = new Readable();
    readable._read = () => {};
    const on = jest.spyOn(readable, 'once');

    sendResponse(socket, 200, {}, readable);

    expect(on).toHaveBeenCalledWith('error', expect.any(Function));
  });

  test('gracefully handles unknown status code', () => {
    sendResponse(socket, 499, { 'Content-Type': 'text/plain' }, 'Weird code');
    expect(socket.write).toHaveBeenCalledWith(expect.stringMatching(/^HTTP\/1.1 499 \r\n/));
  });
});
