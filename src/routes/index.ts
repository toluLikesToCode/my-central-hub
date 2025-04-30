// routes/index.ts
import './stream.routes';
import './files.routes';
import './metrics.routes';
import './file-hosting.routes';
import router from '../core/router';
import { sendResponse } from '../entities/sendResponse';

router.any('/echo', async (req, sock) => {
  let message = 'Hello, world!';
  try {
    const b = req.body ? JSON.parse(req.body.toString()) : {};
    if (typeof b?.message === 'string') message = b.message;
  } catch {
    /* empty */
  }
  const res = JSON.stringify({ message });
  sendResponse(
    sock,
    200,
    {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(res).toString(),
    },
    res,
  );
});

export {}; // side-effect imports run immediately
