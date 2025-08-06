# Embedding Service API Reference

## Overview

The Embedding Service API V2 is a high-performance FastAPI-based service that generates CLIP embeddings for images and videos. It supports batch processing, multiple source types including Google Cloud Storage, and provides optimized GPU-accelerated inference.

**Base URL**: `http://localhost:8080` (configurable via `PYTHON_PORT`)  
**WebSocket URL**: `ws://localhost:8080` (configurable via `PYTHON_PORT`)  
**API Version**: V2 (Batching + WebSocket)  
**Content-Type**: `application/json`

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check the service health and status.

#### Response

**Status Code**: `200 OK`

```json
{
  "status": "ok",
  "uptime_seconds": 12345.67,
  "processed_items_count": 1000,
  "gpu_available": true,
  "model_loaded": true,
  "model_name": "openai/clip-vit-base-patch32",
  "device": "cuda:0",
  "request_queue_size": 0
}
```

#### Response Fields

| Field                   | Type           | Description                                                    |
| ----------------------- | -------------- | -------------------------------------------------------------- |
| `status`                | string         | Service status: `"ok"` or `"error_model_not_loaded"`           |
| `uptime_seconds`        | number         | Service uptime in seconds                                      |
| `processed_items_count` | integer        | Total number of items processed since startup                  |
| `gpu_available`         | boolean        | Whether GPU is available (CUDA or MPS)                         |
| `model_loaded`          | boolean        | Whether CLIP model is loaded and ready                         |
| `model_name`            | string \| null | Name of the loaded CLIP model                                  |
| `device`                | string \| null | Device being used (e.g., "cuda:0", "cpu", "mps")               |
| `request_queue_size`    | integer        | Current number of items in processing queue (HTTP + WebSocket) |

---

### 2. WebSocket Embedding Generation ⚡ **No Timeout Limits**

**WebSocket** `/ws/embed`

Generate CLIP embeddings using persistent WebSocket connection. **Identical API to HTTP endpoint** but eliminates timeout issues for long-running requests.

#### Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/embed');
```

#### Message Protocol

All communication uses JSON messages with the following structure:

##### Client Request Message

```json
{
  "type": "embed_batch",
  "data": {
    "items": [...],
    "request_id": "optional-request-id",
    "isDryRun": false
  },
  "message_id": "optional-tracking-id"
}
```

##### Server Response Message

**Success Response:**

```json
{
  "type": "embed_batch_result",
  "data": {
    "results": [...],
    "batch_id": "batch-12345",
    "processed_by_request_id": "req-67890"
  },
  "message_id": "optional-tracking-id"
}
```

**Error Response:**

```json
{
  "type": "error",
  "error": "Error message description",
  "message_id": "optional-tracking-id"
}
```

#### Message Schema

##### WebSocketRequest

| Field        | Type                    | Required | Description                                     |
| ------------ | ----------------------- | -------- | ----------------------------------------------- |
| `type`       | `string`                | Yes      | Must be `"embed_batch"`                         |
| `data`       | `BatchEmbeddingRequest` | Yes      | Same as HTTP endpoint request body              |
| `message_id` | `string \| null`        | No       | Optional ID for tracking request/response pairs |

##### WebSocketResponse

| Field        | Type                             | Required | Description                                      |
| ------------ | -------------------------------- | -------- | ------------------------------------------------ |
| `type`       | `string`                         | Yes      | `"embed_batch_result"` or `"error"`              |
| `data`       | `BatchEmbeddingResponse \| null` | No       | Response data (null for error responses)         |
| `error`      | `string \| null`                 | No       | Error message (only present for error responses) |
| `message_id` | `string \| null`                 | No       | Echoed message ID from request                   |

#### WebSocket Connection Lifecycle

1. **Connect**: Client establishes WebSocket connection
2. **Send**: Client sends JSON request message
3. **Receive**: Server processes request and sends JSON response
4. **Repeat**: Connection stays open for multiple requests
5. **Disconnect**: Either side can close connection

#### Error Handling

**Connection Errors:**

- Connection refused: Service not running
- Connection closed: Network issues or server shutdown
- Invalid JSON: Malformed message format
- Validation errors: Invalid message schema

**Processing Errors:**
Same error types as HTTP endpoint, returned in WebSocket error responses.

#### Usage Examples

##### Python WebSocket Client

```python
import asyncio
import json
import websockets

async def embed_with_websocket(items, ws_url="ws://localhost:8080/ws/embed"):
    async with websockets.connect(ws_url) as websocket:
        # Send request
        request = {
            "type": "embed_batch",
            "data": {"items": items},
            "message_id": "req-123"
        }
        await websocket.send(json.dumps(request))

        # Receive response
        response = await websocket.recv()
        data = json.loads(response)

        if data["type"] == "error":
            raise Exception(data["error"])

        return data["data"]

# Example usage
items = [
    {
        "id": "img-1",
        "media_type": "image",
        "source_type": "gcs_blob",
        "source": "photos/beach.jpg"
    }
]

results = await embed_with_websocket(items)
```

##### JavaScript WebSocket Client

```javascript
class EmbeddingWebSocketClient {
  constructor(wsUrl = 'ws://localhost:8080/ws/embed') {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.pendingRequests = new Map();
  }

  async connect() {
    this.ws = new WebSocket(this.wsUrl);

    return new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve();
      this.ws.onerror = (error) => reject(error);

      this.ws.onmessage = (event) => {
        const response = JSON.parse(event.data);
        const messageId = response.message_id;

        if (messageId && this.pendingRequests.has(messageId)) {
          const { resolve, reject } = this.pendingRequests.get(messageId);
          this.pendingRequests.delete(messageId);

          if (response.type === 'error') {
            reject(new Error(response.error));
          } else {
            resolve(response.data);
          }
        }
      };
    });
  }

  async embedBatch(batchRequest) {
    const messageId = crypto.randomUUID();

    const wsRequest = {
      type: 'embed_batch',
      data: batchRequest,
      message_id: messageId,
    };

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(messageId, { resolve, reject });
      this.ws.send(JSON.stringify(wsRequest));
    });
  }
}

// Usage
const client = new EmbeddingWebSocketClient();
await client.connect();

const results = await client.embedBatch({
  items: [
    {
      id: 'img-1',
      media_type: 'image',
      source_type: 'gcs_blob',
      source: 'photos/sunset.jpg',
    },
  ],
});
```

#### WebSocket vs HTTP Comparison

| Feature                   | HTTP `/api/embed_batch`       | WebSocket `/ws/embed`        |
| ------------------------- | ----------------------------- | ---------------------------- |
| **Request Format**        | ✅ Same                       | ✅ Same (wrapped in message) |
| **Response Format**       | ✅ Same                       | ✅ Same (wrapped in message) |
| **Processing Logic**      | ✅ Identical                  | ✅ Identical                 |
| **Timeout Limits**        | ❌ HTTP client timeouts       | ✅ No timeout limits         |
| **Connection Overhead**   | ❌ New connection per request | ✅ Persistent connection     |
| **Long-running Requests** | ❌ May timeout                | ✅ Handles any duration      |
| **Multiple Requests**     | ❌ New connection each time   | ✅ Reuse same connection     |
| **Error Handling**        | ✅ HTTP status codes          | ✅ Message-level errors      |
| **Browser Compatibility** | ✅ Universal support          | ✅ Modern browser support    |

---

### 3. HTTP Batch Embedding Generation

**POST** `/api/embed_batch`

Generate CLIP embeddings for a batch of media items (images and/or videos).

#### Request Body

```json
{
  "items": [
    {
      "id": "unique-item-id",
      "media_type": "image|video",
      "source_type": "url|filepath|gcs_blob|buffer_id",
      "source": "source-location",
      "num_frames": 20,
      "original_filename": "optional-filename.jpg"
    }
  ],
  "request_id": "optional-request-id",
  "isDryRun": false
}
```

#### Request Headers

| Header         | Required | Description                                                        |
| -------------- | -------- | ------------------------------------------------------------------ |
| `Content-Type` | Yes      | Must be `application/json`                                         |
| `X-Request-ID` | No       | Optional request ID for tracking (overridden by body `request_id`) |

#### Request Schema

##### BatchEmbeddingRequest

| Field        | Type                        | Required | Description                                                         |
| ------------ | --------------------------- | -------- | ------------------------------------------------------------------- |
| `items`      | `MediaItem[]`               | Yes      | Array of media items to process                                     |
| `request_id` | `string \| null`            | No       | Optional request ID for tracking                                    |
| `isDryRun`   | `boolean \| string \| null` | No       | If true, skips embedding generation and frame extraction timestamps |

##### MediaItem

| Field               | Type              | Required | Description                                                        |
| ------------------- | ----------------- | -------- | ------------------------------------------------------------------ |
| `id`                | `string`          | Yes      | Unique identifier for this media item                              |
| `media_type`        | `string`          | Yes      | Media type: `"image"` or `"video"`                                 |
| `source_type`       | `string`          | Yes      | Source type: `"url"`, `"filepath"`, `"gcs_blob"`, or `"buffer_id"` |
| `source`            | `string`          | Yes      | Source location (see Source Types below)                           |
| `num_frames`        | `integer \| null` | No       | Number of frames to extract from video (default: 20)               |
| `original_filename` | `string \| null`  | No       | Original filename for logging/debugging                            |

#### Source Types

##### URL (`"url"`)

Download media from a public HTTP/HTTPS URL.

```json
{
  "source_type": "url",
  "source": "https://example.com/image.jpg"
}
```

##### File Path (`"filepath"`)

Access media from local filesystem or mounted volume.

```json
{
  "source_type": "filepath",
  "source": "/public/images/photo.jpg"
}
```

- Paths starting with `/public/` are treated as relative to `PYTHON_MEDIA_ROOT`
- Absolute paths are used as-is
- Relative paths are resolved against `PYTHON_MEDIA_ROOT`

##### Google Cloud Storage (`"gcs_blob"`) ⚡ **High Performance**

Access media directly from Google Cloud Storage.

```json
{
  "source_type": "gcs_blob",
  "source": "images/vacation/beach-sunset.jpg"
}
```

- `source` is the exact GCS object name/path
- Bucket configured via `GCS_BUCKET_NAME` environment variable
- Supports parallel batch downloads for optimal performance

##### Buffer ID (`"buffer_id"`)

_Currently not implemented_ - for future multipart upload support.

#### Response

**Status Code**: `200 OK`

```json
{
  "results": [
    {
      "id": "unique-item-id",
      "embedding": [0.1, -0.2, 0.3, ...],
      "error": null,
      "detail": null,
      "debugMetadata": {
        "original_item_id": "unique-item-id",
        "source_type": "gcs_blob",
        "processing_time_ms": 150.5,
        "image_dimensions": "1920x1080",
        "gcs_blob_name": "images/photo.jpg"
      }
    }
  ],
  "batch_id": "batch-12345",
  "processed_by_request_id": "req-67890"
}
```

**Status Code**: `503 Service Unavailable`

```json
{
  "detail": "Model not ready or failed to load."
}
```

#### Response Schema

##### BatchEmbeddingResponse

| Field                     | Type                | Description                              |
| ------------------------- | ------------------- | ---------------------------------------- |
| `results`                 | `EmbeddingResult[]` | Array of embedding results for each item |
| `batch_id`                | `string`            | Internal batch processing ID             |
| `processed_by_request_id` | `string \| null`    | Request ID used for processing           |

##### EmbeddingResult

| Field           | Type               | Description                                      |
| --------------- | ------------------ | ------------------------------------------------ |
| `id`            | `string`           | Item ID from request                             |
| `embedding`     | `number[] \| null` | CLIP embedding vector (typically 512 dimensions) |
| `error`         | `string \| null`   | Error message if processing failed               |
| `detail`        | `string \| null`   | Detailed error information                       |
| `debugMetadata` | `object \| null`   | Debug information about processing               |

#### Debug Metadata Fields

Common debug metadata fields include:

| Field                           | Type      | Description                               |
| ------------------------------- | --------- | ----------------------------------------- |
| `original_item_id`              | `string`  | Original item ID                          |
| `source_type`                   | `string`  | Source type used                          |
| `processing_time_ms`            | `number`  | Processing time in milliseconds           |
| `image_dimensions`              | `string`  | Image dimensions (e.g., "1920x1080")      |
| `num_extracted_frames_for_item` | `integer` | Number of frames extracted (videos)       |
| `gcs_blob_name`                 | `string`  | GCS blob name (for GCS sources)           |
| `resolved_filepath`             | `string`  | Resolved file path (for filepath sources) |
| `downloaded_url`                | `string`  | Downloaded URL (for URL sources)          |
| `frame_extraction_method`       | `string`  | Method used for frame extraction          |
| `video_duration_s`              | `number`  | Video duration in seconds                 |

---

## Examples

### WebSocket Examples

#### Single Image via WebSocket

```javascript
// JavaScript WebSocket client
const ws = new WebSocket('ws://localhost:8080/ws/embed');

ws.onopen = () => {
  const request = {
    type: 'embed_batch',
    data: {
      items: [
        {
          id: 'img-001',
          media_type: 'image',
          source_type: 'gcs_blob',
          source: 'photos/beach-sunset.jpg',
          original_filename: 'beach-sunset.jpg',
        },
      ],
      request_id: 'single-image-ws-001',
    },
    message_id: 'msg-001',
  };

  ws.send(JSON.stringify(request));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  if (response.type === 'embed_batch_result') {
    console.log('Embedding received:', response.data.results[0].embedding);
  }
};
```

#### Python WebSocket with Multiple Requests

```python
import asyncio
import json
import websockets

async def process_multiple_batches():
    uri = "ws://localhost:8080/ws/embed"

    async with websockets.connect(uri) as websocket:
        # Process multiple batches on same connection
        batches = [
            {
                "items": [{"id": f"img-{i}", "media_type": "image",
                          "source_type": "gcs_blob", "source": f"batch1/img{i}.jpg"}],
                "request_id": f"batch-{i}"
            }
            for i in range(5)
        ]

        for i, batch in enumerate(batches):
            # Send request
            request = {
                "type": "embed_batch",
                "data": batch,
                "message_id": f"msg-{i}"
            }
            await websocket.send(json.dumps(request))

            # Receive response
            response_raw = await websocket.recv()
            response = json.loads(response_raw)

            print(f"Batch {i} processed: {len(response['data']['results'])} results")

# Run
asyncio.run(process_multiple_batches())
```

### HTTP Examples

#### Single Image from GCS

```bash
curl -X POST "http://localhost:8080/api/embed_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "id": "img-001",
        "media_type": "image",
        "source_type": "gcs_blob",
        "source": "photos/beach-sunset.jpg",
        "original_filename": "beach-sunset.jpg"
      }
    ],
    "request_id": "single-image-001"
  }'
```

### Batch Processing Mixed Media

```bash
curl -X POST "http://localhost:8080/api/embed_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "id": "img-001",
        "media_type": "image",
        "source_type": "gcs_blob",
        "source": "images/photo1.jpg"
      },
      {
        "id": "vid-001",
        "media_type": "video",
        "source_type": "url",
        "source": "https://example.com/video.mp4",
        "num_frames": 15
      },
      {
        "id": "img-002",
        "media_type": "image",
        "source_type": "filepath",
        "source": "/public/uploads/image.png"
      }
    ],
    "request_id": "mixed-batch-001"
  }'
```

### Dry Run (No Embeddings)

```bash
curl -X POST "http://localhost:8080/api/embed_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "id": "test-001",
        "media_type": "image",
        "source_type": "gcs_blob",
        "source": "test/sample.jpg"
      }
    ],
    "isDryRun": true
  }'
```

---

## Performance Features

### WebSocket Advantages ⚡

- **No Timeout Limits**: Process videos with 100+ frames without HTTP timeouts
- **Persistent Connections**: Eliminate connection overhead for multiple requests
- **Real-time Processing**: Immediate response delivery without HTTP handshake delays
- **Connection Reuse**: Process thousands of items on single connection
- **Bi-directional**: Server can send status updates or progress notifications (future feature)

### Batch Processing

- Multiple items processed concurrently
- GPU batch inference for optimal throughput
- Dynamic batching based on VRAM availability

### GCS Optimization ⚡

- **2-4x faster** than URL downloads for images
- Parallel downloads for multiple items
- Direct streaming (no temp files for images)
- Connection pooling and retry handling

### Video Processing

- Hardware-accelerated frame extraction (CUDA, VAAPI, etc.)
- Intelligent frame sampling (entropy-based, scene detection)
- Configurable frame count per video

### Memory Management

- Streaming downloads to minimize memory usage
- Dynamic VRAM management for optimal GPU utilization
- Automatic cleanup of temporary files

---

## Configuration

### Environment Variables

| Variable                          | Default                        | Description                           |
| --------------------------------- | ------------------------------ | ------------------------------------- |
| `PYTHON_PORT`                     | `8080`                         | Server port                           |
| `PYTHON_MEDIA_ROOT`               | `/app/public`                  | Root path for filepath sources        |
| `CLIP_MODEL`                      | `openai/clip-vit-base-patch32` | CLIP model to load                    |
| `LOG_LEVEL`                       | `INFO`                         | Logging level                         |
| `TARGET_VRAM_UTILIZATION`         | `0.80`                         | Target GPU VRAM utilization (0.0-1.0) |
| `MAX_BATCH_ITEMS`                 | `128`                          | Maximum items per GPU batch           |
| `BATCH_FLUSH_TIMEOUT_S`           | `0.5`                          | Batch flush timeout in seconds        |
| `DEFAULT_VIDEO_FRAMES_TO_EXTRACT` | `20`                           | Default video frame count             |
| `GCS_BUCKET_NAME`                 | `media-content-1`              | GCS bucket name                       |
| `GCS_MAX_WORKERS`                 | `8`                            | Concurrent GCS download workers       |
| `GCS_TIMEOUT`                     | `60`                           | GCS request timeout in seconds        |
| `FFMPEG_HWACCEL_METHOD`           | `cuda`                         | FFmpeg hardware acceleration method   |

---

## Error Handling

### Common Error Responses

#### Model Not Ready (503)

```json
{
  "detail": "Model not ready or failed to load."
}
```

#### Individual Item Errors (200 with error in result)

```json
{
  "results": [
    {
      "id": "failed-item",
      "embedding": null,
      "error": "Failed to download GCS blob",
      "detail": "Blob not found: missing/file.jpg",
      "debugMetadata": {...}
    }
  ]
}
```

### Error Types

| Error                                  | Description             | Resolution                                   |
| -------------------------------------- | ----------------------- | -------------------------------------------- |
| `"Model not ready or failed to load."` | CLIP model not loaded   | Check model configuration and logs           |
| `"Failed to download GCS blob"`        | GCS access failed       | Verify blob exists and credentials are valid |
| `"Filepath not found"`                 | File not accessible     | Check file path and permissions              |
| `"Failed to download URL"`             | HTTP download failed    | Verify URL is accessible                     |
| `"Unsupported source_type"`            | Invalid source type     | Use supported source types                   |
| `"Failed to extract frames"`           | Video processing failed | Check video format and FFmpeg availability   |

---

## Rate Limits & Quotas

- **Concurrent Requests**: Limited by `request_queue_size` (shared between HTTP and WebSocket)
- **Batch Size**: Maximum items per request limited by `MAX_BATCH_ITEMS`
- **VRAM**: Dynamic batching based on available GPU memory
- **GCS**: Limited by `GCS_MAX_WORKERS` concurrent downloads
- **WebSocket Connections**: No explicit limit (bounded by system resources)

---

## SDK Examples

### WebSocket SDKs

#### Python WebSocket Client Class

```python
import asyncio
import json
import uuid
import websockets
from typing import Dict, Any, List

class EmbeddingWebSocketClient:
    def __init__(self, ws_url: str = "ws://localhost:8080/ws/embed"):
        self.ws_url = ws_url
        self.websocket = None
        self.pending_requests: Dict[str, asyncio.Future] = {}

    async def connect(self):
        self.websocket = await websockets.connect(self.ws_url)
        asyncio.create_task(self._listen_for_responses())

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()

    async def _listen_for_responses(self):
        try:
            async for message in self.websocket:
                response = json.loads(message)
                message_id = response.get("message_id")

                if message_id and message_id in self.pending_requests:
                    future = self.pending_requests.pop(message_id)

                    if response["type"] == "error":
                        future.set_exception(Exception(response["error"]))
                    else:
                        future.set_result(response["data"])
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")

    async def embed_batch(self, batch_request: Dict[str, Any]) -> Dict[str, Any]:
        if not self.websocket:
            raise Exception("WebSocket not connected. Call connect() first.")

        message_id = str(uuid.uuid4())

        ws_request = {
            "type": "embed_batch",
            "data": batch_request,
            "message_id": message_id
        }

        future = asyncio.Future()
        self.pending_requests[message_id] = future

        await self.websocket.send(json.dumps(ws_request))
        return await future

# Usage
client = EmbeddingWebSocketClient()
await client.connect()

results = await client.embed_batch({
    "items": [
        {
            "id": "img-1",
            "media_type": "image",
            "source_type": "gcs_blob",
            "source": "photos/vacation.jpg"
        }
    ]
})

embeddings = [r["embedding"] for r in results["results"] if r["embedding"]]
await client.disconnect()
```

#### JavaScript WebSocket Client Class

```javascript
class EmbeddingWebSocketClient {
  constructor(wsUrl = 'ws://localhost:8080/ws/embed') {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.pendingRequests = new Map();
  }

  async connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => resolve();
      this.ws.onerror = (error) => reject(error);

      this.ws.onmessage = (event) => {
        const response = JSON.parse(event.data);
        const messageId = response.message_id;

        if (messageId && this.pendingRequests.has(messageId)) {
          const { resolve, reject } = this.pendingRequests.get(messageId);
          this.pendingRequests.delete(messageId);

          if (response.type === 'error') {
            reject(new Error(response.error));
          } else {
            resolve(response.data);
          }
        }
      };
    });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }

  async embedBatch(batchRequest) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected. Call connect() first.');
    }

    const messageId = crypto.randomUUID();

    const wsRequest = {
      type: 'embed_batch',
      data: batchRequest,
      message_id: messageId,
    };

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(messageId, { resolve, reject });
      this.ws.send(JSON.stringify(wsRequest));
    });
  }
}

// Usage
const client = new EmbeddingWebSocketClient();
await client.connect();

const results = await client.embedBatch({
  items: [
    {
      id: 'img-1',
      media_type: 'image',
      source_type: 'gcs_blob',
      source: 'photos/vacation.jpg',
    },
  ],
});

const embeddings = results.results.filter((r) => r.embedding).map((r) => r.embedding);

client.disconnect();
```

### HTTP SDKs

#### Python

```python
import requests

def embed_batch(items, base_url="http://localhost:8080"):
    response = requests.post(
        f"{base_url}/api/embed_batch",
        json={"items": items},
        timeout=300
    )
    return response.json()

# Example usage
items = [
    {
        "id": "img-1",
        "media_type": "image",
        "source_type": "gcs_blob",
        "source": "photos/vacation.jpg"
    }
]

results = embed_batch(items)
embeddings = [r["embedding"] for r in results["results"] if r["embedding"]]
```

### JavaScript/Node.js

```javascript
async function embedBatch(items, baseUrl = 'http://localhost:8080') {
  const response = await fetch(`${baseUrl}/api/embed_batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  });
  return await response.json();
}

// Example usage
const items = [
  {
    id: 'img-1',
    media_type: 'image',
    source_type: 'gcs_blob',
    source: 'photos/vacation.jpg',
  },
];

const results = await embedBatch(items);
const embeddings = results.results.filter((r) => r.embedding).map((r) => r.embedding);
```

---

## Performance Benchmarks

#### HTTP vs WebSocket Response Times

| Scenario                   | Items | Source | HTTP Time     | WebSocket Time | WebSocket Advantage |
| -------------------------- | ----- | ------ | ------------- | -------------- | ------------------- |
| Single image               | 1     | GCS    | 150-400ms     | 110-300ms      | ~25% faster         |
| Single image               | 1     | URL    | 600-1800ms    | 500-1500ms     | ~20% faster         |
| Batch images               | 10    | GCS    | 900-1400ms    | 800-1200ms     | ~15% faster         |
| Large video (50 frames)    | 1     | GCS    | **Timeout**   | 8000-15000ms   | **No timeout**      |
| Multiple batches (10 reqs) | 50    | GCS    | 15000-25000ms | 8000-12000ms   | ~50% faster         |

#### Connection Overhead Comparison

| Metric                      | HTTP           | WebSocket     | Improvement                  |
| --------------------------- | -------------- | ------------- | ---------------------------- |
| **Connection Setup**        | Per request    | Once          | N/A                          |
| **TCP Handshakes**          | Every request  | One-time      | ~50-100ms saved per request  |
| **TLS Handshakes**          | Every request  | One-time      | ~100-200ms saved per request |
| **Request Headers**         | ~200-500 bytes | ~50-100 bytes | ~75% reduction               |
| **Multiple Requests (10x)** | 10 connections | 1 connection  | ~90% less overhead           |

#### Memory Usage

- **Base model**: ~2GB VRAM (CLIP ViT-B/32)
- **Per image**: ~10MB VRAM
- **Per video frame**: ~10MB VRAM
- **Batch overhead**: ~100MB VRAM
- **WebSocket overhead**: ~1-5MB RAM per connection

#### Timeout Scenarios

| Request Type                 | HTTP Timeout Risk | WebSocket Timeout | Notes                       |
| ---------------------------- | ----------------- | ----------------- | --------------------------- |
| Single image                 | ❌ Low            | ✅ None           | Fast processing             |
| Small video (≤20 frames)     | ⚠️ Medium         | ✅ None           | Usually under 30s           |
| Large video (50+ frames)     | ❌ **High**       | ✅ None           | Often exceeds HTTP timeouts |
| Batch processing (20+ items) | ❌ **Very High**  | ✅ None           | Cumulative processing time  |
| Network-slow sources         | ❌ **Very High**  | ✅ None           | Download time varies        |

---

_Last updated: August 6, 2025_
