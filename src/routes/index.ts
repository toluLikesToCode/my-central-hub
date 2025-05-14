/* eslint-disable @typescript-eslint/no-explicit-any */
// routes/index.ts
import './stream.routes';
import './files.routes';
import './metrics.routes';
import './embeddings.routes';
import './file-hosting.routes';
import router from '../core/router';
import { sendWithContext } from '../entities/sendResponse';

// Extracts the message from request body or returns default
function extractMessage(body?: Buffer): string {
  if (!body) return 'Hello, world!';
  let parsed: unknown;
  try {
    parsed = JSON.parse(body.toString());
  } catch {
    // Invalid JSON; return default message
    return 'Hello, world!';
  }
  if (parsed && typeof (parsed as any).message === 'string') {
    return (parsed as any).message;
  }
  return 'Hello, world!';
}

// Define routes with their message extraction strategy
const routes = [
  { path: '/ping', getMessage: (): string | undefined => undefined },
  { path: '/echo', getMessage: extractMessage },
];

// Register handlers for each route
routes.forEach(({ path, getMessage }) =>
  router.any(path, async (req, sock) => {
    const message = getMessage(req.body as Buffer | undefined);
    const responseText = message ? JSON.stringify({ message }) : '';
    sendWithContext(
      req,
      sock,
      200,
      {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(responseText).toString(),
      },
      responseText,
    );
  }),
);

export {}; // side-effect imports run immediately
