import { parser } from '../../src/core/parser';

describe('HTTP Parser', () => {
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

    expect(parsed.method).toBeUndefined();
    expect(parsed.path).toBeUndefined();
  });
});
