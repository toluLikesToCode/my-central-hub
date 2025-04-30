// file-streamer/fileService.ts
/**
 * @deprecated This file is now obsolete after moving to file-hosting module.
 * This file is deprecated and should be removed. All file management logic is now in file-hosting/fileHostingService.ts.
 */
export class FileService {
  constructor(private readonly rootDir: string) {
    throw new Error('FileService is deprecated. Use file-hosting module instead.');
  }
}
