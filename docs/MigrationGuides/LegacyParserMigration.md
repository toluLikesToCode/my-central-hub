# Legacy Parser Migration Guide

## Background

The original `parser.parse()` function in `src/core/parser.ts` has been deprecated in favor of the more robust `HttpRequestParser` class in `src/core/httpParser.ts`. This document provides guidance on how to migrate from the legacy parser to the new implementation.

## Why Migrate?

The new `HttpRequestParser` provides several advantages:

1. Better support for HTTP pipelining (multiple requests in a single connection)
2. Support for chunked encoding
3. Proper handling of binary content
4. More efficient memory usage
5. Better error reporting

## Migration Steps

### 1. Import the new parser

```typescript
// Old import
import { parser } from '../core/parser';

// New import
import { HttpRequestParser } from '../core/httpParser';
```

### 2. Create a parser instance

The new parser is a class that must be instantiated:

```typescript
// Create a parser instance - typically you'd do this once per connection
const parser = new HttpRequestParser();
```

### 3. Feeding data

```typescript
// Old approach (synchronous, string-based)
const request = parser.parse(rawStringData);

// New approach (accepts buffers, may return null if request isn't complete)
const request = parser.feed(bufferData);
if (request) {
  // Process the request
}
```

### 4. Checking for incomplete requests

```typescript
// Check if there are pending bytes that haven't formed a complete request
const pendingBytes = parser.getPendingBytes();
if (pendingBytes > 0) {
  // There's an incomplete request - wait for more data
}
```

### 5. Handling pipelined requests

```typescript
// Process any complete requests that were parsed
let req = parser.feed(chunk);
while (req) {
  // Process the request
  await processRequest(req);

  // Check for more complete requests in the buffer
  req = parser.feed(Buffer.alloc(0));
}
```

### 6. Resetting the parser

```typescript
// Reset the parser (useful in error conditions)
parser.reset();
```

## Complete Example

```typescript
import { HttpRequestParser } from '../core/httpParser';
import { createServer } from 'net';

const server = createServer((socket) => {
  const parser = new HttpRequestParser();

  socket.on('data', async (chunk) => {
    try {
      // Process any complete requests
      let req = parser.feed(chunk);
      while (req) {
        // Handle the request
        await handleRequest(req, socket);

        // Check for more complete requests in the buffer
        req = parser.feed(Buffer.alloc(0));
      }

      // Check if we have an incomplete request
      const pendingBytes = parser.getPendingBytes();
      if (pendingBytes > MAX_PENDING_BYTES) {
        // Too much pending data, potential attack
        socket.end();
      }
    } catch (err) {
      // Handle parsing errors
      console.error('Parser error:', err);
      socket.end();
    }
  });
});
```

## Testing

All tests for the legacy parser have been marked as deprecated. New tests should use the `HttpRequestParser` class instead. See `tests/core/httpParser.test.ts` for examples.
