import { extname } from 'path';
import { mimeTypes } from './mimeTypes';

export function getMimeType(fileName: string): string {
  const ext = extname(fileName).toLowerCase();
  return mimeTypes[ext] || 'application/octet-stream';
}
