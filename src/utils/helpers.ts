import { extname } from 'path';
import { mimeTypes } from './mimeTypes';

export function getMimeType(fileName: string): string {
  // ccheck if its a file name or a path
  // if its a path, get the file name
  const fileParts = fileName.split('/');
  const lastPart = fileParts[fileParts.length - 1];

  // check if the last part has a file name
  if (lastPart.includes('.')) {
    fileName = lastPart;
  } else {
    // if its a directory, return the default mime type
    return 'application/octet-stream';
  }
  const ext = extname(fileName).toLowerCase();
  return mimeTypes[ext] || 'application/octet-stream';
}
