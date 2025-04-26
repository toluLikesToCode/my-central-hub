// src/entities/http.ts
export interface IncomingRequest {
  method: string;
  path: string;
  httpVersion: string;
  headers: Record<string, string>;
  body?: Buffer;
  raw: string;
  query: Record<string, string>;
  /** set by parser when the request-line or headers are malformed */
  invalid?: boolean;

  /** Fully-resolved URL built from the request line (optional for back-compat) */
  url?: URL;

  /** Canonical multi-value header map (keys are lower-cased) */
  headersMap?: Map<string, string[]>;

  /** Bag for connection-scoped or auth metadata */
  ctx?: Record<string, unknown>;
}
