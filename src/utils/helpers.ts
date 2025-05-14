import { extname } from 'path';
import { mimeTypes } from './mimeTypes';

/**
 * Returns the MIME type for a given file name or file path.
 *
 * If the input is a directory or does not contain a file extension,
 * the function returns 'application/octet-stream' as the default MIME type.
 * Otherwise, it extracts the file extension and returns the corresponding MIME type
 * from the `mimeTypes` mapping. If the extension is not recognized,
 * it also defaults to 'application/octet-stream'.
 *
 * @param fileName - The file name or file path to determine the MIME type for.
 * @returns The MIME type as a string.
 */
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

/**
 * Compares two strings in a case-insensitive manner using ASCII order.
 *
 * Converts both input strings to lowercase and compares them.
 * Returns -1 if `a` comes before `b`, 1 if `a` comes after `b`, or 0 if they are equal.
 *
 * @param a - The first string to compare.
 * @param b - The second string to compare.
 * @returns A number indicating the sort order: -1, 0, or 1.
 */
export function nocaseAscii(a: string, b: string): number {
  const aa = a.toLowerCase();
  const bb = b.toLowerCase();
  return aa < bb ? -1 : aa > bb ? 1 : 0; // exact byte order after case-fold
}
