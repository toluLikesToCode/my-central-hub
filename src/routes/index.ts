// routes/index.ts
import './stream.routes';
import './files.routes';
import './metrics.routes';
import './embeddings.routes';
import './file-hosting.routes';
import router from '../core/router';
import { sendResponse } from '../entities/sendResponse';

['/echo', '/ping'].forEach((path) =>
  router.any(path, async (req, sock) => {
    /**
     * Determines the message to be used based on the request path and body.
     *
     * - If the path is `/ping`, the message is set to `undefined`.
     * - Otherwise, it attempts to parse the request body as JSON and extract the `message` property.
     * - If the `message` property exists and is a string, it is returned.
     * - If parsing fails or the `message` property is not a string, a default message of `'Hello, world!'` is returned.
     *
     * @constant
     * @type {string | undefined}
     */
    const message =
      path === '/ping'
        ? undefined
        : (() => {
            try {
              const body = req.body ? JSON.parse(req.body.toString()) : {};
              return typeof body?.message === 'string' ? body.message : 'Hello, world!';
            } catch {
              return 'Hello, world!';
            }
          })();
    const res = message ? JSON.stringify({ message }) : '';
    sendResponse(
      sock,
      200,
      { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(res).toString() },
      res,
    );
  }),
);

export {}; // side-effect imports run immediately
