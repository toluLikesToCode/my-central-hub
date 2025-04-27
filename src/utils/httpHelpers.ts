// Simple header accessor that falls back to legacy Record<string,string>
import { IncomingRequest } from '../entities/http';

/**
 * Case-insensitive lookup that handles multi-value headers.
 * @param req - The incoming request object.
 * @param name - The name of the header to retrieve.
 * @returns The value of the header, or undefined if not found.
 * @example
 * const contentType = getHeader(req, 'Content-Type');
 * const userAgent = getHeader(req, 'User-Agent');
 * const customHeader = getHeader(req, 'X-Custom-Header');
 */
export function getHeader(req: IncomingRequest, name: string): string | undefined {
  const key = name.toLowerCase();
  if (req.headersMap && req.headersMap.has(key)) {
    return req.headersMap.get(key)![0]; // first value
  }
  return req.headers?.[key];
}

/** Case-sensitive query lookup that falls back to searchParams */
export function getQuery(req: IncomingRequest, key: string): string | undefined {
  if (req.invalid || !req.query || !req.url) return undefined;

  // ✅ look in the parser-built map first
  const direct = req.query?.[key];
  if (direct !== undefined) return direct;

  // fallback (rare) – parse from URL
  return req.url.searchParams.get(key) ?? undefined;
}
