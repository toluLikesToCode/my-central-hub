// src/core/parser.ts
import { IncomingRequest } from '../entities/http';
import { URL } from 'url'; // new – used for req.url below

const ALLOWED_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'] as const;

// helper to accumulate duplicate headers without losing the original Record<string,string>
function addHeader(map: Map<string, string[]>, key: string, value: string) {
  const k = key.toLowerCase();
  const arr = map.get(k) ?? [];
  arr.push(value);
  map.set(k, arr);
}

export const parser = {
  parse(raw: string): IncomingRequest {
    // --- quick sanity guard ---
    if (raw.length === 0) {
      return { method: '', path: '', httpVersion: '', headers: {}, raw, query: {}, invalid: true };
    }

    const [head = '', bodyString = ''] = raw.split('\r\n\r\n');
    const lines = head.split('\r\n');

    // --- request-line validation ---
    const [requestLine, ...headerLines] = lines;
    const parts = requestLine.split(' ');
    if (parts.length < 3) {
      return {
        method: '',
        path: '',
        httpVersion: '',
        headers: {},
        headersMap: new Map(),
        raw,
        query: {},
        ctx: {},
        invalid: true,
      };
    }

    const [method, fullPath, httpVersion] = parts;

    let invalid =
      !method ||
      !ALLOWED_METHODS.includes(method as any) ||
      !fullPath.startsWith('/') ||
      !httpVersion.startsWith('HTTP/');

    // --- split URI + query ---
    const [path, queryString] = fullPath.split('?');
    const query: Record<string, string> = {};
    (queryString ?? '').split('&').forEach((p) => {
      if (!p) return;
      const [k, v = ''] = p.split('=');
      query[decodeURIComponent(k)] = decodeURIComponent(v);
    });

    // --- headers ---
    const headers: Record<string, string> = {};
    const headersMap = new Map<string, string[]>();

    const MAX_HEADERS = 1000;
    if (headerLines.length > MAX_HEADERS) invalid = true;

    for (const line of headerLines) {
      const idx = line.indexOf(':');
      if (idx === -1) {
        invalid = true;
        continue;
      }
      const key = line.slice(0, idx).trim();
      const value = line.slice(idx + 1).trim();
      headers[key.toLowerCase()] = value; // legacy access
      addHeader(headersMap, key, value); // future-proof
    }

    const body = bodyString ? Buffer.from(bodyString, 'utf-8') : undefined;

    return {
      method,
      path,
      httpVersion,
      headers,
      headersMap,
      body,
      raw,
      query,
      ctx: {}, // reserved for later
      url: new URL(fullPath, 'http://placeholder'), // absolute for now
      invalid,
    };
  },
};
