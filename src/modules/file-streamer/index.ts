import { IncomingRequest } from '../../entities/http';
import { Socket } from 'net';
import { FileService } from './fileService';
import { URL } from 'url';

export const fileStreamingController = {
  async handleStream(req: IncomingRequest, socket: Socket): Promise<void> {
    try {
      if (!req.path) {
        socket.write('HTTP/1.1 400 Bad Request\r\n\r\nMissing path');
        socket.end();
        return;
      }
      const url = new URL(req.path, `http://${req.headers.host}`);
      const fileName = url.searchParams.get('file');

      if (!fileName) {
        socket.write('HTTP/1.1 400 Bad Request\r\n\r\nMissing file parameter');
        socket.end();
        return;
      }

      const range = req.headers['range'];

      FileService.streamFile(fileName, range, socket);
    } catch {
      socket.write('HTTP/1.1 500 Internal Server Error\r\n\r\nServer Error');
      socket.end();
    }
  },

  async listFiles(req: IncomingRequest, socket: Socket): Promise<void> {
    try {
      const filesJson = FileService.listFiles();
      const responseBody = JSON.stringify(filesJson);

      const headers = [
        'HTTP/1.1 200 OK',
        'Content-Type: application/json',
        `Content-Length: ${responseBody.length}`,
        '',
        '',
      ].join('\r\n');

      socket.write(headers);
      socket.write(responseBody);
      socket.end();
    } catch {
      socket.write('HTTP/1.1 500 Internal Server Error\r\n\r\nServer Error');
      socket.end();
    }
  },
};
