/**
 * @deprecated This parser implementation is deprecated and will be removed in a future version.
 * Use the HttpRequestParser class from httpParser.ts instead which provides better support
 * for HTTP pipelining, chunked encoding, and binary content.
 *
 * Migration guide:
 * 1. Import HttpRequestParser: import { HttpRequestParser } from './httpParser';
 * 2. Create an instance: const parser = new HttpRequestParser();
 * 3. For each chunk of data: const request = parser.feed(Buffer.from(data));
 * 4. If request is null, more data is needed; if not null, a complete request is available
 * 5. For multiple requests in one connection (pipelining): process all with req = parser.feed(Buffer.alloc(0))
 */

import { IncomingRequest } from '../entities/http';
import { URL } from 'url';

const ALLOWED_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'] as const;
const MAX_HEADERS = 1000;

// helper for duplicate headers
function addHeader(map: Map<string, string[]>, key: string, value: string) {
  const k = key.toLowerCase();
  const list = map.get(k) ?? [];
  list.push(value);
  map.set(k, list);
}

export const parser = {
  /**
   * @deprecated Use HttpRequestParser from httpParser.ts instead
   */
  parse(raw: string): IncomingRequest {
    const dummyUrl = new URL('http://placeholder/');
    const earlyReturn = (): IncomingRequest => ({
      url: dummyUrl,
      path: '',
      query: {},
      httpVersion: '',
      method: '',
      headers: {},
      headersMap: new Map(),
      raw,
      ctx: {},
      invalid: true,
    });

    /* -------- empty buffer guard -------- */
    if (raw.length === 0) return earlyReturn();

    const [head = '', bodyString = ''] = raw.split('\r\n\r\n');
    const lines = head.split('\r\n');

    /* -------- request-line split -------- */
    const [requestLine, ...headerLines] = lines;
    const parts = requestLine.split(' ');
    if (parts.length < 3) return earlyReturn();

    const [method, fullPath, httpVersion] = parts;
    const isOptionsStar = method === 'OPTIONS' && fullPath === '*';
    let invalid =
      !ALLOWED_METHODS.includes(method as (typeof ALLOWED_METHODS)[number]) ||
      (!fullPath.startsWith('/') && !isOptionsStar) ||
      !httpVersion.startsWith('HTTP/');

    /* -------- URL + query -------- */
    const url = isOptionsStar
      ? new URL('http://coolcat.com')
      : new URL(fullPath, 'http://coolcat.com'); // Fixed URL creation for non-options requests
    const query: Record<string, string> = {};
    url.searchParams.forEach((v, k) => (query[k] = v));

    /* -------- headers -------- */
    const headers: Record<string, string> = {};
    const headersMap = new Map<string, string[]>();

    if (headerLines.length > MAX_HEADERS) invalid = true;

    for (const line of headerLines) {
      const idx = line.indexOf(':');
      if (idx === -1) {
        invalid = true;
        continue;
      }
      const key = line.slice(0, idx).trim();
      const value = line.slice(idx + 1).trim();
      headers[key.toLowerCase()] = value;
      addHeader(headersMap, key, value);
    }

    const body = bodyString ? Buffer.from(bodyString, 'utf-8') : undefined;

    return {
      url,
      path: isOptionsStar ? '*' : decodeURIComponent(url.pathname),
      query,
      httpVersion,
      method,
      headers,
      headersMap,
      body,
      raw,
      ctx: {},
      invalid,
    };
  },
};
