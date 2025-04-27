import { Socket } from 'net';
import { Readable } from 'stream';

const STATUS_TEXT: Record<number, string> = {
  200: 'OK',
  206: 'Partial Content',
  400: 'Bad Request',
  404: 'Not Found',
  405: 'Method Not Allowed',
  416: 'Range Not Satisfiable', // ← new
  500: 'Internal Server Error',
};

export function sendResponse(
  socket: Socket,
  status: number,
  headers: Record<string, string>,
  body?: string | Buffer | Readable,
): void {
  const head =
    `HTTP/1.1 ${status} ${STATUS_TEXT[status] ?? ''}\r\n` +
    Object.entries(headers)
      .map(([k, v]) => `${k}: ${v}`)
      .join('\r\n') +
    '\r\n\r\n';

  socket.write(head);

  if (!body) {
    socket.end();
    return;
  }

  if (body instanceof Readable) {
    body.pipe(socket);
    body.once('error', () => socket.end());
    body.once('end', () => socket.end());
  } else {
    socket.write(body);
    socket.end();
  }
}
