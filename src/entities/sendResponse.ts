import { Socket } from 'net';

export function sendResponse(
  socket: Socket,
  statusCode: number,
  headers: Record<string, string>,
  body: string | Buffer,
): void {
  const statusMessage = getStatusMessage(statusCode);

  const headerLines = Object.entries(headers)
    .map(([key, value]) => `${key}: ${value}`)
    .join('\r\n');

  const responseHeaders = `HTTP/1.1 ${statusCode} ${statusMessage}\r\n${headerLines}\r\n\r\n`;

  socket.write(responseHeaders);
  socket.write(body);
  socket.end();
}

function getStatusMessage(statusCode: number): string {
  switch (statusCode) {
    case 200:
      return 'OK';
    case 201:
      return 'Created';
    case 204:
      return 'No Content';
    case 206:
      return 'Partial Content';
    case 400:
      return 'Bad Request';
    case 401:
      return 'Unauthorized';
    case 403:
      return 'Forbidden';
    case 404:
      return 'Not Found';
    case 416:
      return 'Range Not Satisfiable';
    case 500:
      return 'Internal Server Error';
    default:
      return 'Unknown';
  }
}
