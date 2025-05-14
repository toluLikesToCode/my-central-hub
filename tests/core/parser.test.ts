/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

/**
 * @deprecated These tests are for the deprecated legacy parser implementation.
 * New tests should be written for HttpRequestParser from httpParser.ts instead.
 * This file will be removed in a future version.
 */

import { parser } from '../../src/core/parser';

describe.skip('HTTP Parser (DEPRECATED)', () => {
  // NOTE: All tests are skipped because they test deprecated
  // functionality that will be removed
  it('should parse a simple GET request', () => {
    const raw = 'GET /hello HTTP/1.1\r\nHost: localhost\r\nUser-Agent: test\r\n\r\n';
    const parsed = parser.parse(raw);

    expect(parsed.method).toBe('GET');
    expect(parsed.path).toBe('/hello');
    expect(parsed.httpVersion).toBe('HTTP/1.1');
    expect(parsed.headers.host).toBe('localhost');
    expect(parsed.headers['user-agent']).toBe('test');
  });

  it('should handle missing headers gracefully', () => {
    const raw = 'POST /upload HTTP/1.1\r\n\r\n';
    const parsed = parser.parse(raw);

    expect(parsed.method).toBe('POST');
    expect(parsed.path).toBe('/upload');
    expect(parsed.headers).toEqual({});
  });

  it('should not crash on completely malformed request', () => {
    const raw = 'INVALID REQUEST';
    const parsed = parser.parse(raw);

    expect(parsed.invalid).toBe(true);
    expect(parsed.method).toBeFalsy();
    expect(parsed.path).toBeFalsy();
  });

  it('should parse url.pathname and headersMap correctly', () => {
    const raw = 'GET /hello HTTP/1.1\r\nHost: localhost\r\nUser-Agent: test\r\n\r\n';
    const parsed = parser.parse(raw);

    expect(parsed.url?.pathname).toBe('/hello');
    expect(parsed.headersMap?.get('host')).toEqual(['localhost']);
  });

  it('exposes url, path, and query consistently', () => {
    const raw = 'GET /stream?file=test.mp4 HTTP/1.1\r\nHost: x\r\n\r\n';
    const r = parser.parse(raw);
    expect(r.url.pathname).toBe('/stream');
    expect(r.path).toBe('/stream');
    expect(r.query).toEqual({ file: 'test.mp4' });
  });

  it('includes httpVersion', () => {
    const raw = 'GET /test HTTP/1.1\r\nHost: x\r\n\r\n';
    const r = parser.parse(raw);
    expect(r.httpVersion).toBe('HTTP/1.1');
  });

  it('should handle completely empty request', () => {
    const raw = '';
    const parsed = parser.parse(raw);

    expect(parsed.invalid).toBe(true);
    expect(parsed.method).toBeFalsy();
    expect(parsed.path).toBeFalsy();
    expect(parsed.headers).toEqual({});
  });

  it('should correctly parse multiple query parameters', () => {
    const raw = 'GET /search?q=nodejs&sort=desc HTTP/1.1\r\nHost: localhost\r\n\r\n';
    const parsed = parser.parse(raw);
    expect(parsed.method).toBe('GET');
    expect(parsed.path).toBe('/search');
    expect(parsed.httpVersion).toBe('HTTP/1.1');
    expect(parsed.headers.host).toBe('localhost');
    expect(parsed.query).toEqual({ q: 'nodejs', sort: 'desc' });
  });

  it('should decode percent-encoded paths', () => {
    const raw = 'GET /foo%20bar HTTP/1.1\r\nHost: localhost\r\n\r\n';
    const parsed = parser.parse(raw);
    expect(parsed.path).toBe('/foo bar');
  });

  it('should handle OPTIONS * request', () => {
    const raw = 'OPTIONS * HTTP/1.1\r\nHost: localhost\r\n\r\n';
    const parsed = parser.parse(raw);
    expect(parsed.method).toBe('OPTIONS');
    expect(parsed.path).toBe('*');
  });

  it('should handle duplicated headers gracefully', () => {
    const raw = `GET / HTTP/1.1\r\nHost: localhost\r\nCookie: a=1\r\nCookie: b=2\r\n\r\n`;
    const parsed = parser.parse(raw);

    expect(parsed.headers.host).toBe('localhost');
    expect(parsed.headers['cookie']).toBe('b=2'); // Note: last wins in simple parsing
    expect(parsed.headersMap?.get('cookie')).toEqual(['a=1', 'b=2']);
  });
});
