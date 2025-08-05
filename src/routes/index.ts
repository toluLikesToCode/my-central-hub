// routes/index.ts
import './stream.routes';
import './files.routes';
import './metrics.routes';
import './embeddings.routes';
import './file-hosting.routes';
import './logging.routes';
import router, { Handler } from '../core/router';
import { sendWithContext } from '../entities/sendResponse';
import { createContext, runInContext } from 'vm';

// True echo handler that mirrors the request body exactly
// Special behavior: executes JavaScript code when content-type is application/javascript
const echoHandler: Handler = async (req, sock) => {
  const body = req.body as Buffer | undefined;

  // If no body, return empty response
  if (!body || body.length === 0) {
    sendWithContext(
      req,
      sock,
      200,
      {
        'Content-Type': 'text/plain',
        'Content-Length': '0',
      },
      '',
    );
    return;
  }

  // Determine content type from request headers or infer from body
  let contentType = req.headers['content-type'] || 'application/octet-stream';

  // If no content-type header, try to infer intelligently
  if (!req.headers['content-type']) {
    const bodyStr = body.toString('utf8');

    // Try to detect JSON
    if (bodyStr.trim().startsWith('{') || bodyStr.trim().startsWith('[')) {
      try {
        JSON.parse(bodyStr);
        contentType = 'application/json';
      } catch {
        // Not valid JSON, keep as octet-stream
      }
    }
    // Try to detect XML
    else if (bodyStr.trim().startsWith('<')) {
      contentType = 'application/xml';
    }
    // Check if it's valid UTF-8 text
    else if (Buffer.compare(body, Buffer.from(bodyStr, 'utf8')) === 0) {
      contentType = 'text/plain';
    }
  }

  // Special handling for JavaScript execution
  if (contentType === 'application/javascript' || contentType === 'text/javascript') {
    const code = body.toString('utf8');

    try {
      // Create a more comprehensive sandbox with Node.js modules
      const sandbox = {
        console,
        setTimeout,
        setInterval,
        clearTimeout,
        clearInterval,
        Promise,
        Buffer,
        Math,
        Date,
        JSON,
        String,
        Number,
        Array,
        Object,
        RegExp,
        Error,
        // Add Node.js modules
        require: (moduleName: string) => {
          const allowedModules = ['https', 'http', 'url', 'crypto', 'querystring', 'path'];
          if (allowedModules.includes(moduleName)) {
            // eslint-disable-next-line @typescript-eslint/no-require-imports
            return require(moduleName);
          }
          throw new Error(`Module '${moduleName}' is not allowed`);
        },
        process: {
          version: process.version,
          platform: process.platform,
          env: {}, // Empty for security
        },
      };

      // Wrap the code to capture the result properly
      const wrappedCode = `
        (async () => {
          ${code}
        })()
      `;

      const context = createContext(sandbox);
      const result = runInContext(wrappedCode, context, {
        timeout: 30000, // 30 seconds
        displayErrors: true,
      });

      // Handle the result (which should be a Promise)
      const finalResult = await result;

      if (finalResult && finalResult.binaryData && Buffer.isBuffer(finalResult.binaryData)) {
        // cast finalResult.binaryData as Buffer;
        // Ensure finalResult has contentType for binary data

        const data = finalResult.binaryData as Buffer;

        // Return the binary data directly with proper headers
        sendWithContext(
          req,
          sock,
          200,
          {
            'Content-Type': finalResult.contentType || 'video/mp4',
            'Content-Length': finalResult.contentLength || data.length.toString(),
            'Cache-Control': 'public, max-age=3600',
          },
          data, // Send raw buffer
        );
        return;
      }

      sendWithContext(
        req,
        sock,
        200,
        {
          'Content-Type': 'application/json',
          'Content-Length': JSON.stringify({
            executed: true,
            result: finalResult,
            type: typeof finalResult,
            timestamp: new Date().toISOString(),
          }).length.toString(),
        },
        JSON.stringify({
          executed: true,
          result: finalResult,
          type: typeof finalResult,
          timestamp: new Date().toISOString(),
        }),
      );
      return;
    } catch (error: unknown) {
      // Handle execution errors
      const errorResponse = {
        executed: false,
        error: error instanceof Error ? error.message : 'Unknown execution error',
        type: 'ExecutionError',
        timestamp: new Date().toISOString(),
      };

      const responseBody = JSON.stringify(errorResponse, null, 2);

      sendWithContext(
        req,
        sock,
        400,
        {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(responseBody).toString(),
        },
        responseBody,
      );
      return;
    }
  }

  // Default echo behavior for non-JavaScript content
  sendWithContext(
    req,
    sock,
    200,
    {
      'Content-Type': contentType,
      'Content-Length': body.length.toString(),
    },
    body,
  );
};

// Register ping route
router.any('/ping', async (req, sock) => {
  sendWithContext(
    req,
    sock,
    200,
    {
      'Content-Type': 'application/json',
      'Content-Length': '0',
    },
    '',
  );
});

// Register echo route with true echo behavior
router.any('/echo', echoHandler);

export {}; // side-effect imports run immediately
