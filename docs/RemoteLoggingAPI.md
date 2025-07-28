# Remote Logging API Documentation

This document describes how to use the remote logging API endpoint for sending log entries from client applications to the server.

## Overview

The remote logging API allows client applications to send structured log entries to a central server for storage and analysis. This is particularly useful for collecting logs from distributed applications or client-side applications running in browsers.

## API Endpoint

- **URL**: `/api/logs/ingest`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Authorization**: Bearer token or API key (optional)

## Authentication

If authentication is configured on the server, you must include an `Authorization` header with your requests:

```
Authorization: Bearer <your-token>
```

or

```
Authorization: ApiKey <your-api-key>
```

## Request Payload

The request body should be a JSON array containing one or more log entry objects. Each log entry must conform to the following schema:

```typescript
interface LogEntry {
  level: string; // Required: Log level (e.g., 'info', 'warn', 'error')
  message: any; // Required: The log message (string or object)
  timestamp: string; // Required: ISO 8601 formatted date string
  contextName?: string; // Optional: Context or category name
  meta?: {
    // Optional: Additional metadata
    [key: string]: any;
  };
  error?: {
    // Optional: Error information (for error logs)
    name?: string; // Optional: Error type name
    message: string; // Required for error objects: Error message
    stack?: string; // Optional: Stack trace
    [key: string]: any; // Other custom error properties
  };
}
```

### Example Request Payload

```json
[
  {
    "level": "info",
    "message": "Application started",
    "contextName": "GalleryGenerator",
    "timestamp": "2025-05-17T12:00:00.000Z",
    "meta": {
      "version": "1.2.0",
      "environment": "production"
    }
  },
  {
    "level": "error",
    "message": "Failed to connect to database",
    "contextName": "DatabaseService",
    "timestamp": "2025-05-17T12:01:30.000Z",
    "meta": {
      "retry": true,
      "host": "db1"
    },
    "error": {
      "name": "ConnectionError",
      "message": "Timeout occurred",
      "stack": "Error: Timeout occurred\n    at connectToDatabase (dbService.js:45:10)",
      "code": "ETIMEDOUT"
    }
  }
]
```

## Response Format

### Success Response (200 OK)

```json
{
  "status": "ok",
  "stored": 2
}
```

### Partial Success Response (207 Multi-Status)

If some entries were successfully stored but others were invalid:

```json
{
  "status": "partial",
  "stored": 1,
  "invalid": 1,
  "invalidEntries": [1]
}
```

### Error Responses

- **400 Bad Request**: Invalid request format or all log entries were invalid
- **401 Unauthorized**: Authentication failed
- **500 Internal Server Error**: Server error processing the request

## Client Integration Examples

See the following example files for client integration:

- JavaScript Example: `/examples/client-log-example.js`
- TypeScript Example: `/examples/client-log-example.ts`

## Best Practices

1. **Batch Log Entries**: Send multiple log entries in a single request to reduce network overhead.

2. **Include Context**: Always include a `contextName` to categorize logs.

3. **Structured Metadata**: Use the `meta` object for structured data rather than embedding it in the message.

4. **Error Details**: For error logs, include comprehensive details in the `error` object.

5. **Log Levels**: Use appropriate log levels:

   - `debug`: Detailed information for debugging
   - `info`: General information about application operation
   - `warn`: Warning conditions that might lead to errors
   - `error`: Error conditions that affect operation but don't stop the application
   - `fatal`: Severe errors that stop the application

6. **Timestamps**: Use ISO 8601 format for timestamps for consistency.

## Server-Side Storage

Logs are stored in an SQLite database in the following schema:

```sql
CREATE TABLE logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  level TEXT NOT NULL,
  message TEXT NOT NULL,
  contextName TEXT,
  meta TEXT,
  timestamp TEXT NOT NULL,
  error_name TEXT,
  error_message TEXT,
  error_stack TEXT,
  error_details TEXT,
  client_ip TEXT,
  user_agent TEXT,
  received_at TEXT NOT NULL
);
```
