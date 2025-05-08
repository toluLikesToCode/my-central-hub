import { IncomingRequest } from '../entities/http';
import { URL } from 'url';
import logger from '../utils/logger';

enum ParserState {
  REQUEST_LINE,
  HEADERS,
  BODY,
  CHUNK_SIZE,
  CHUNK_BODY,
  CHUNK_TRAILER,
  DONE,
  ERROR,
}

const ALLOWED_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'];
const MAX_HEADER_BYTES = 8192; // 8KB
const MAX_HEADERS = 100; // Maximum number of headers allowed
const MAX_BODY_BYTES = 10 * 1024 * 1024; // 10 MB
const CRLF = Buffer.from('\r\n');

export class HttpRequestParser {
  protected buffer = Buffer.alloc(0);
  private state = ParserState.REQUEST_LINE;
  private headers: Record<string, string> = {};
  private headersMap = new Map<string, string[]>();
  private bodyChunks: Buffer[] = [];
  private method = '';
  private httpVersion = '';
  private url!: URL;
  private contentLength = 0;
  private remainingBody = 0;
  private isChunked = false;
  private invalid = false;
  private lastHeaderKey: string | null = null;
  private isBinaryContent = false;

  /**
   * Returns the number of pending bytes in the parser buffer.
   */
  public getPendingBytes(): number {
    return this.buffer.length;
  }

  feed(data: Buffer): IncomingRequest | null {
    this.buffer = Buffer.concat([this.buffer, data]);
    try {
      while (true) {
        // REQUEST_LINE
        if (this.state === ParserState.REQUEST_LINE) {
          const idx = this.buffer.indexOf('\r\n');
          if (idx === -1) return null;
          const requestLine = this.buffer.subarray(0, idx).toString('utf8');
          this.buffer = this.buffer.subarray(idx + 2);
          const parts = requestLine.split(' ');
          if (parts.length !== 3) {
            this._setError('Invalid request line: ' + requestLine);
            return this._errorResponse();
          }
          const [method, reqPath, version] = parts;
          if (!ALLOWED_METHODS.includes(method)) {
            this._setError('Unsupported method: ' + method);
            continue;
          }
          if (!version.startsWith('HTTP/')) {
            this._setError('Invalid HTTP version: ' + version);
            continue;
          }
          this.method = method;
          this.httpVersion = version;
          try {
            this.url = new URL(reqPath, 'http://placeholder');
          } catch {
            this._setError('Malformed URL: ' + reqPath);
            continue;
          }
          this.state = ParserState.HEADERS;
        }
        // HEADERS
        if (this.state === ParserState.HEADERS) {
          // find end-of-headers or handle empty header block
          const idx = this.buffer.indexOf('\r\n\r\n');
          let headersRaw = '';
          if (idx === -1) {
            // immediate blank line → zero headers
            if (this.buffer.subarray(0, 2).equals(CRLF)) {
              this.buffer = this.buffer.subarray(2);
            } else {
              if (this.buffer.length > MAX_HEADER_BYTES) {
                this._setError('Headers too large');
                return this._errorResponse();
              }
              return null;
            }
          } else {
            headersRaw = this.buffer.subarray(0, idx).toString('utf8');
            this.buffer = this.buffer.subarray(idx + 4);
          }
          const lines = headersRaw.split('\r\n');
          let headerCount = 0;
          for (const line of lines) {
            if (line.trim() === '') continue;
            // support folded headers per RFC7230 §3.2.4
            if (line.startsWith(' ') || line.startsWith('\t')) {
              if (this.lastHeaderKey) {
                const prev = this.headers[this.lastHeaderKey];
                this.headers[this.lastHeaderKey] = prev + ' ' + line.trim();
                // update map too
                this.headersMap.set(
                  this.lastHeaderKey,
                  (this.headersMap.get(this.lastHeaderKey) || []).concat(
                    this.headers[this.lastHeaderKey],
                  ),
                );
                continue;
              } else {
                this._setError('Invalid header folding');
                return this._errorResponse();
              }
            }
            const colon = line.indexOf(':');
            if (colon === -1) {
              this._setError('Invalid header line: ' + line);
              continue;
            }
            const key = line.slice(0, colon).trim().toLowerCase();
            const value = line.slice(colon + 1).trim();
            this.headers[key] = value;
            this.headersMap.set(key, [...(this.headersMap.get(key) ?? []), value]);
            this.lastHeaderKey = key;
            headerCount++;
            if (headerCount > MAX_HEADERS) {
              this._setError('Too many headers');
              return this._errorResponse();
            }
          }

          // Check for binary content
          const contentType = this.headers['content-type'] || '';
          this.isBinaryContent =
            contentType.includes('application/octet-stream') ||
            contentType.includes('image/') ||
            contentType.includes('video/') ||
            contentType.includes('audio/') ||
            contentType.includes('multipart/form-data');

          // Short-circuit on parse errors
          if (this.invalid) {
            return this._errorResponse();
          }
          // enforce Host header for HTTP/1.1
          if (this.httpVersion === 'HTTP/1.1' && !this.headers['host']) {
            this._setError('Missing Host header');
            return this._errorResponse();
          }
          if (
            this.headers['transfer-encoding'] &&
            this.headers['transfer-encoding'].toLowerCase() === 'chunked'
          ) {
            this.isChunked = true;
            this.state = ParserState.CHUNK_SIZE;
          } else if (this.headers['content-length']) {
            this.contentLength = parseInt(this.headers['content-length'], 10);
            if (isNaN(this.contentLength) || this.contentLength < 0) {
              this._setError('Invalid Content-Length');
              continue;
            }
            this.remainingBody = this.contentLength;
            this.state = this.contentLength > 0 ? ParserState.BODY : ParserState.DONE;
          } else {
            this.state = ParserState.DONE;
          }
        }
        // BODY
        if (this.state === ParserState.BODY) {
          this.contentLength = parseInt(this.headers['content-length'], 10);
          if (isNaN(this.contentLength) || this.contentLength < 0) {
            this._setError('Invalid Content-Length');
            continue;
          }
          if (this.contentLength > MAX_BODY_BYTES) {
            this._setError('Request body too large');
            continue;
          }
          this.remainingBody = this.contentLength;
          if (this.buffer.length < this.remainingBody) return null;
          this.bodyChunks.push(this.buffer.subarray(0, this.remainingBody));
          this.buffer = this.buffer.subarray(this.remainingBody);
          this.remainingBody = 0;
          this.state = ParserState.DONE;
        }
        // CHUNK_SIZE
        if (this.state === ParserState.CHUNK_SIZE) {
          const idx = this.buffer.indexOf('\r\n');
          if (idx === -1) return null;
          const line = this.buffer.subarray(0, idx).toString('utf8');
          this.buffer = this.buffer.subarray(idx + 2);
          const chunkSize = parseInt(line, 16);
          if (isNaN(chunkSize)) {
            this._setError('Invalid chunk size: ' + line);
            continue;
          }
          if (chunkSize === 0) {
            this.state = ParserState.CHUNK_TRAILER;
          } else {
            this.remainingBody = chunkSize;
            this.state = ParserState.CHUNK_BODY;
          }
        }
        // CHUNK_BODY
        if (this.state === ParserState.CHUNK_BODY) {
          if (this.remainingBody > MAX_BODY_BYTES) {
            this._setError('Chunk body too large');
            continue;
          }
          if (this.buffer.length < this.remainingBody) return null;
          const chunk = this.buffer.subarray(0, this.remainingBody);
          this.bodyChunks.push(chunk);
          this.buffer = this.buffer.subarray(this.remainingBody);
          this.remainingBody = 0;
          if (this.buffer.length < 2 || !this.buffer.subarray(0, 2).equals(CRLF)) {
            this._setError('Missing CRLF after chunk');
            continue;
          }
          this.buffer = this.buffer.subarray(2);
          this.state = ParserState.CHUNK_SIZE;
        }
        // CHUNK_TRAILER
        if (this.state === ParserState.CHUNK_TRAILER) {
          if (this.buffer.length === 0 || this.buffer.equals(CRLF)) {
            this.buffer = Buffer.alloc(0);
            this.state = ParserState.DONE;
            continue;
          }
          const idx = this.buffer.indexOf('\r\n\r\n');
          if (idx === -1) return null;
          this.buffer = this.buffer.subarray(idx + 4);
          this.state = ParserState.DONE;
        }
        // DONE
        if (this.state === ParserState.DONE) {
          // capture leftover before reset (for pipelining)
          const leftover = this.buffer;
          const finalBody = this.bodyChunks.length ? Buffer.concat(this.bodyChunks) : undefined;

          // Log information about binary content for debugging
          if (this.isBinaryContent && finalBody) {
            logger.debug('Parsed binary request', {
              method: this.method,
              contentType: this.headers['content-type'],
              contentLength: finalBody.length,
              path: this.url.pathname,
            });
          }

          const request = {
            method: this.method,
            path: this.url.pathname,
            query: Object.fromEntries(this.url.searchParams.entries()),
            headers: this.headers,
            headersMap: this.headersMap,
            httpVersion: this.httpVersion,
            url: this.url,
            body: finalBody,
            raw: '',
            ctx: {
              isBinaryContent: this.isBinaryContent,
            },
            invalid: this.invalid,
          };
          this.reset();
          this.buffer = leftover; // restore leftover for next request
          return request;
        }
        // ERROR
        if (this.state === ParserState.ERROR) {
          const errReq = this._errorResponse();
          this.reset();
          return errReq;
        }
      }
    } catch (err) {
      this._setError(`Error during parsing: ${(err as Error).message}`);
      const errReq = this._errorResponse();
      this.reset();
      return errReq;
    }
  }

  private _setError(message: string): void {
    logger.error(message);
    this.invalid = true;
    this.state = ParserState.ERROR;
  }

  /** Build a minimal invalid IncomingRequest */
  private _errorResponse(): IncomingRequest {
    return {
      method: '',
      path: '',
      query: {},
      headers: {},
      headersMap: new Map(),
      httpVersion: '',
      url: new URL('http://invalid'),
      body: undefined,
      raw: '',
      ctx: {},
      invalid: true,
    };
  }

  reset(): void {
    this.buffer = Buffer.alloc(0);
    this.state = ParserState.REQUEST_LINE;
    this.headers = {};
    this.headersMap = new Map();
    this.bodyChunks = [];
    this.method = '';
    this.httpVersion = '';
    this.url = new URL('http://placeholder');
    this.contentLength = 0;
    this.remainingBody = 0;
    this.isChunked = false;
    this.invalid = false;
    this.lastHeaderKey = null;
    this.isBinaryContent = false;
  }
}
