# Guide: Implementing a Logging HTTP Endpoint for `HttpProtocolHandler`

This guide is for **server developers** who need to implement an HTTP endpoint that receives log entries from clients using the `HttpProtocolHandler` in a TypeScript logging system.

---

## 1. Overview

Your endpoint will receive **HTTP POST** requests with a JSON array of structured log entries. Each entry follows a strict schema (see below) and may include error and metadata information.

- **HTTP Method:** `POST`
- **Content-Type:** `application/json`
- **Body:** Array of log entry objects (see [LogEntry Schema](#logentry-schema))
- **Authentication:** Bearer token or API key via the `Authorization` header this is optional and may not be included

---

## 2. LogEntry Schema

Below is the TypeScript interface for a log entry, as sent by the client:

```typescript
/**
 * Represents a structured log entry in the logging system.
 *
 * @remarks
 * LogEntry objects are sent as part of a JSON array in the request body.
 *
 * @interface LogEntry
 *
 * @property {string} level - The severity level of the log entry (e.g., 'info', 'warn', 'error').
 * @property {any} message - The main content or message of the log entry.
 * @property {string} [contextName] - Optional name providing context about where the log was generated.
 * @property {Record<string, any>} [meta] - Optional metadata associated with the log entry.
 * @property {string} timestamp - The date and time when the log entry was created (ISO 8601 string).
 * @property {Object} [error] - Optional standardized error information when logging errors.
 * @property {string} [error.name] - The name of the error.
 * @property {string} error.message - The error message.
 * @property {string} [error.stack] - The error stack trace.
 * @property {any} [error.[key: string]] - Additional custom error properties.
 */
export interface LogEntry {
  level: string;
  message: any;
  contextName?: string;
  meta?: Record<string, any>;
  timestamp: string; // ISO 8601
  error?: {
    name?: string;
    message: string;
    stack?: string;
    [key: string]: any;
  };
}
```

### Example JSON Payload

```json
[
  {
    "level": "error",
    "message": "Failed to connect to database",
    "contextName": "DatabaseService",
    "timestamp": "2024-06-01T12:00:00.000Z",
    "meta": { "retry": true, "host": "db1" },
    "error": {
      "name": "ConnectionError",
      "message": "Timeout occurred",
      "stack": "Error: Timeout occurred\n    at ...",
      "code": "ETIMEDOUT"
    }
  },
  {
    "level": "info",
    "message": "Server started",
    "timestamp": "2024-06-01T12:05:00.000Z"
  }
]
```

---

## 3. Authentication

Clients will send an `Authorization` header:

- **Bearer token:**  
  `Authorization: Bearer <token>`
- **API key:**  
  `Authorization: ApiKey <key>`

If the header is present **You must validate this header** according to your security requirements.

---

## 4. Example Express.js Endpoint

Below is a sample implementation using Express.js:

```typescript
import express, { Request, Response } from 'express';

/**
 * Handles incoming log batches from HttpProtocolHandler.
 *
 * @param req - The HTTP request object.
 * @param res - The HTTP response object.
 * @remarks
 * Expects a JSON array of log entries in the request body.
 * Validates authentication and stores logs as needed.
 */
const logIngestHandler = async (req: Request, res: Response) => {
  // 1. Authenticate
  const authHeader = req.header('authorization');
  if (!authHeader || !isValidAuth(authHeader)) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  // 2. Parse and validate body
  if (!Array.isArray(req.body)) {
    return res.status(400).json({ error: 'Expected an array of log entries' });
  }

  // 3. Validate and process each log entry
  for (const entry of req.body) {
    if (
      typeof entry.level !== 'string' ||
      typeof entry.message === 'undefined' ||
      typeof entry.timestamp !== 'string'
    ) {
      return res.status(400).json({ error: 'Invalid log entry format' });
    }
    // Store the log entry (e.g., database, file, etc.)
    await storeLogEntry(entry);
  }

  // 4. Respond with success
  res.status(200).json({ status: 'ok' });
};

// Express app setup
const app = express();
app.use(express.json());
app.post('/ingest', logIngestHandler);
app.listen(3000, () => console.log('Log endpoint listening on port 3000'));

/**
 * Validates the Authorization header.
 * Replace with your actual authentication logic.
 */
function isValidAuth(authHeader: string): boolean {
  // Example: Accept a hardcoded token for demonstration
  return authHeader === 'Bearer YOUR_JWT_TOKEN' || authHeader === 'ApiKey YOUR_API_KEY';
}

/**
 * Stores a log entry.
 * Replace with your actual storage logic (e.g., database, file, etc.).
 */
async function storeLogEntry(entry: any): Promise<void> {
  // Example: Print to console
  console.log('Received log entry:', entry);
}
```

---

## 5. Best Practices

- **Validate** all incoming log entries for required fields.
- **Store** logs in a durable system (database, file, or log aggregator).
- **Respond** with HTTP 200 on success, 4xx on client errors, 5xx on server errors.
- **Secure** your endpoint with authentication and rate limiting.
- **Monitor** for failed or malformed requests.

---

## 6. References

- [Express.js Request Handling](https://expressjs.com/en/guide/routing.html)
- [TSDoc Specification](https://tsdoc.org/)
- [TypeScript Fetch API docs](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
