export interface IncomingRequest {
  method: string | undefined;
  path: string | undefined;
  httpVersion: string | undefined;
  headers: Record<string, string>;
  raw: string;
}
