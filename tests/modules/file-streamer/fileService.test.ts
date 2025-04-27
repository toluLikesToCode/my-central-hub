import { FileService } from '../../../src/modules/file-streamer/fileService';
import * as fs from 'fs';
import { Socket } from 'net';

jest.mock('fs');

describe('FileService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('listFiles', () => {
    it('should list files when media directory exists', () => {
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.readdirSync as jest.Mock).mockReturnValue(['video.mp4', 'audio.mp3']);

      const result = FileService.listFiles();

      expect(result.files).toEqual(['video.mp4', 'audio.mp3']);
    });

    it("should return empty list if media directory doesn't exist", () => {
      (fs.existsSync as jest.Mock).mockReturnValue(false);

      const result = FileService.listFiles();

      expect(result.files).toEqual([]);
    });
  });

  describe('streamFile', () => {
    const fakeSocket = {
      write: jest.fn(),
      end: jest.fn(),
      on: jest.fn(),
      pipe: jest.fn(),
    } as unknown as Socket;

    it('should handle non-existing file gracefully', () => {
      (fs.existsSync as jest.Mock).mockReturnValue(false);

      FileService.streamFile('nofile.mp4', undefined, fakeSocket);

      expect(fakeSocket.write).toHaveBeenCalledWith(expect.stringContaining('404 Not Found'));
      expect(fakeSocket.end).toHaveBeenCalled();
    });

    it('should handle invalid range requests', () => {
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 1000 });

      FileService.streamFile('video.mp4', 'bytes=2000-3000', fakeSocket);

      expect(fakeSocket.write).toHaveBeenCalledWith(
        expect.stringContaining('416 Range Not Satisfiable'),
      );
      expect(fakeSocket.end).toHaveBeenCalled();
    });

    it('should start a stream for a valid file', () => {
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 1000 });
      (fs.createReadStream as jest.Mock).mockReturnValue({
        pipe: jest.fn(),
        on: jest.fn(),
      });

      FileService.streamFile('video.mp4', undefined, fakeSocket);

      expect(fakeSocket.write).toHaveBeenCalledWith(
        expect.stringContaining('HTTP/1.1 206 Partial Content'),
      );
    });

    it('should handle file stream error', () => {
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 1000 });

      const mockStream = {
        pipe: jest.fn(),
        on: jest.fn((event, handler) => {
          if (event === 'error') {
            handler(new Error('Stream failed'));
          }
        }),
      };
      (fs.createReadStream as jest.Mock).mockReturnValue(mockStream);

      FileService.streamFile('video.mp4', undefined, fakeSocket);

      expect(fakeSocket.end).toHaveBeenCalled();
    });

    it('should stream partial content for valid small range', () => {
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 1000 });
      (fs.createReadStream as jest.Mock).mockReturnValue({
        pipe: jest.fn(),
        on: jest.fn(),
      });

      FileService.streamFile('video.mp4', 'bytes=0-499', fakeSocket);

      expect(fakeSocket.write).toHaveBeenCalledWith(
        expect.stringContaining('HTTP/1.1 206 Partial Content'),
      );
    });

    it('should stream partial content for valid range bytes=0-499', () => {
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 1000 });
      (fs.createReadStream as jest.Mock).mockReturnValue({
        pipe: jest.fn(),
        on: jest.fn(),
      });

      FileService.streamFile('video.mp4', 'bytes=0-499', fakeSocket);

      expect(fakeSocket.write).toHaveBeenCalledWith(
        expect.stringContaining('HTTP/1.1 206 Partial Content'),
      );
    });

    it('should handle range with no start (e.g., bytes=-500)', () => {
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 1000 });
      (fs.createReadStream as jest.Mock).mockReturnValue({
        pipe: jest.fn(),
        on: jest.fn(),
      });

      FileService.streamFile('video.mp4', 'bytes=-500', fakeSocket);

      expect(fakeSocket.write).toHaveBeenCalledWith(
        expect.stringContaining('HTTP/1.1 206 Partial Content'),
      );
    });

    it('should handle range with no end (e.g., bytes=500-)', () => {
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 1000 });
      (fs.createReadStream as jest.Mock).mockReturnValue({
        pipe: jest.fn(),
        on: jest.fn(),
      });

      FileService.streamFile('video.mp4', 'bytes=500-', fakeSocket);

      expect(fakeSocket.write).toHaveBeenCalledWith(
        expect.stringContaining('HTTP/1.1 206 Partial Content'),
      );
    });

    it('should return 416 for invalid range format (e.g., bytes=invalid)', () => {
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 1000 });

      FileService.streamFile('video.mp4', 'bytes=invalid', fakeSocket);

      expect(fakeSocket.write).toHaveBeenCalledWith(
        expect.stringContaining('416 Range Not Satisfiable'),
      );
      expect(fakeSocket.end).toHaveBeenCalled();
    });
  });
});
