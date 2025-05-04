/**
 * src/entities/http.ts
 * This file contains the interface definitions for HTTP requests.
 */

export interface IncomingRequest {
  url: URL; // canonical URL (always present)
  path: string; // == url.pathname
  query: Record<string, string>; // decoded single-value map
  httpVersion: string; // e.g. "HTTP/1.1"

  method: string;
  headers: Record<string, string>;
  headersMap?: Map<string, string[]>;
  body?: Buffer;
  raw: string;
  ctx?: Record<string, unknown>;
  invalid?: boolean;
}
