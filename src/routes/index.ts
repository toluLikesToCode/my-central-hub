/* eslint-disable @typescript-eslint/no-explicit-any */
// routes/index.ts
import './stream.routes';
import './files.routes';
import './metrics.routes';
import './embeddings.routes';
import './file-hosting.routes';
import './logging.routes';
import router from '../core/router';
import { sendWithContext } from '../entities/sendResponse';

// Extracts the message from request body or returns default
function extractMessage(body?: Buffer): string {
  if (!body) return 'Hello, world!';
  const str = body.toString();
  // Try to parse as JSON first
  try {
    const parsed = JSON.parse(str);
    if (parsed && typeof (parsed as any).message === 'string') {
      return (parsed as any).message;
    }
  } catch {
    // Not JSON, fall through to treat as raw text
  }
  // If not JSON, treat the body as plain text
  if (str.trim().length > 0) {
    return str;
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
