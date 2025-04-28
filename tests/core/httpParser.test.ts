import { HttpRequestParser } from '../../src/core/httpParser'; // Ensure this file exists or update the path
import { IncomingRequest } from '../../src/entities/http'; // Ensure this file exists or update the path

function feedAll(parser: HttpRequestParser, str: string): IncomingRequest | null {
  return parser.feed(Buffer.from(str, 'utf8'));
}

describe('HttpRequestParser', () => {
  let parser: HttpRequestParser;

  beforeEach(() => {
    parser = new HttpRequestParser();
  });

  test('parses simple GET request', () => {
    const req = feedAll(parser, 'GET /hello HTTP/1.1\r\nHost: example.com\r\n\r\n');
    expect(req).not.toBeNull();
    expect(req?.method).toBe('GET');
    expect(req?.path).toBe('/hello');
    expect(req?.headers['host']).toBe('example.com');
    expect(req?.invalid).toBeFalsy();
  });

  test('parses POST with Content-Length', () => {
    const req = feedAll(
      parser,
      'POST /submit HTTP/1.1\r\nHost: test\r\nContent-Length: 11\r\n\r\nHello World',
    );
    expect(req).not.toBeNull();
    expect(req?.method).toBe('POST');
    expect(req?.body?.toString()).toBe('Hello World');
  });

  test('handles chunked transfer-encoding', () => {
    const parts = [
      'POST /upload HTTP/1.1\r\nHost: test\r\nTransfer-Encoding: chunked\r\n\r\n',
      '5\r\nHello\r\n',
      '6\r\n World\r\n',
      '0\r\n\r\n',
    ];
    let result: IncomingRequest | null = null;
    for (const p of parts) {
      const feedResult = parser.feed(Buffer.from(p, 'utf8'));
      if (feedResult) result = feedResult;
    }
    if (!result) {
      const feedResult = parser.feed(Buffer.alloc(0));
      if (feedResult) result = feedResult;
    }
    expect(result).not.toBeNull();
    expect(result?.body?.toString()).toBe('Hello World');
  });

  test('rejects invalid request line', () => {
    const req = feedAll(parser, 'BADREQUEST\r\nHost: test\r\n\r\n');
    expect(req).not.toBeNull();
    expect(req?.invalid).toBeTruthy();
  });

  test('rejects unsupported HTTP method', () => {
    const req = feedAll(parser, 'FOO / HTTP/1.1\r\nHost: test\r\n\r\n');
    expect(req).not.toBeNull();
    expect(req?.invalid).toBeTruthy();
  });

  test('rejects too many headers', () => {
    const headers = Array(1005).fill('X-Test: 123').join('\r\n');
    const req = feedAll(parser, `GET / HTTP/1.1\r\n${headers}\r\n\r\n`);
    expect(req).not.toBeNull();
    expect(req?.invalid).toBeTruthy();
  });

  test('rejects body exceeding MAX_BODY_BYTES', () => {
    const bigBody = 'A'.repeat(11 * 1024 * 1024); // 11MB
    const req = feedAll(
      parser,
      `POST / HTTP/1.1\r\nHost: test\r\nContent-Length: ${bigBody.length}\r\n\r\n${bigBody}`,
    );
    expect(req).not.toBeNull();
    expect(req?.invalid).toBeTruthy();
  });

  test('parses partial requests across multiple TCP chunks', () => {
    const req1 = parser.feed(Buffer.from('GET /mul', 'utf8'));
    expect(req1).toBeNull();

    const req2 = parser.feed(Buffer.from('ti-part HTTP/1.1\r\nHost: te', 'utf8'));
    expect(req2).toBeNull();

    const req3 = parser.feed(Buffer.from('st\r\n\r\n', 'utf8'));
    expect(req3).not.toBeNull();
    expect(req3?.method).toBe('GET');
    expect(req3?.path).toBe('/multi-part');
    expect(req3?.headers['host']).toBe('test');
  });

  test('parses headers with continuations', () => {
    const req = feedAll(
      parser,
      'GET / HTTP/1.1\r\nHost: test\r\nX-Long-Header: abc\r\n def\r\n\r\n',
    );
    expect(req).not.toBeNull();
    expect(req?.headers['x-long-header']).toBe('abc def');
  });

  test('ignores trailers after chunked body', () => {
    const parts = [
      'POST /upload HTTP/1.1\r\nHost: test\r\nTransfer-Encoding: chunked\r\n\r\n',
      '5\r\nHello\r\n',
      '6\r\n World\r\n',
      '0\r\nSome-Trailer: yes\r\n\r\n',
    ];
    let result: IncomingRequest | null = null;
    for (const p of parts) {
      const feedResult = parser.feed(Buffer.from(p, 'utf8'));
      if (feedResult) result = feedResult;
    }
    expect(result).not.toBeNull();
    expect(result?.body?.toString()).toBe('Hello World');
  });

  test('rejects missing Host header in HTTP/1.1', () => {
    const req = feedAll(parser, 'GET /test HTTP/1.1\r\n\r\n');
    expect(req).not.toBeNull();
    expect(req?.invalid).toBeTruthy();
  });

  test('allows HTTP/1.0 without Host header', () => {
    const req = feedAll(parser, 'GET /test HTTP/1.0\r\n\r\n');
    expect(req).not.toBeNull();
    expect(req?.invalid).toBeFalsy();
  });
});
