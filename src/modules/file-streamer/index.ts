import { IncomingRequest } from '../../entities/http';
import { Socket } from 'net';
import { FileService } from './fileService';
import { URL } from 'url';
import { sendResponse } from '../../entities/sendResponse';

export const fileStreamingController = {
  async handleStream(req: IncomingRequest, socket: Socket): Promise<void> {
    try {
      if (!req.path) {
        sendResponse(socket, 400, { 'Content-Type': 'text/plain' }, 'Missing path');
        return;
      }
      const url = new URL(req.path, `http://${req.headers.host}`);
      const fileName = url.searchParams.get('file');

      if (!fileName) {
        sendResponse(socket, 400, { 'Content-Type': 'text/plain' }, 'Missing file parameter');
        return;
      }

      const range = req.headers['range'];

      FileService.streamFile(fileName, range, socket);
    } catch {
      sendResponse(socket, 500, { 'Content-Type': 'text/plain' }, 'Server Error');
    }
  },

  async listFiles(req: IncomingRequest, socket: Socket): Promise<void> {
    try {
      const filesJson = FileService.listFiles();
      const responseBody = JSON.stringify(filesJson);
      sendResponse(
        socket,
        200,
        {
          'Content-Type': 'application/json',
          'Content-Length': responseBody.length.toString(),
        },
        responseBody,
      );
    } catch {
      sendResponse(socket, 500, { 'Content-Type': 'text/plain' }, 'Server Error');
    }
  },
};
