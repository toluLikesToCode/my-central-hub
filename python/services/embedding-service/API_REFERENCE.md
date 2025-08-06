# Embedding Service API Reference

## Overview

The Embedding Service API V2 is a high-performance FastAPI-based service that generates CLIP embeddings for images and videos. It supports batch processing, multiple source types including Google Cloud Storage, and provides optimized GPU-accelerated inference.

**Base URL**: `http://localhost:8080` (configurable via `PYTHON_PORT`)  
**API Version**: V2 (Batching)  
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

| Field                   | Type           | Description                                          |
| ----------------------- | -------------- | ---------------------------------------------------- |
| `status`                | string         | Service status: `"ok"` or `"error_model_not_loaded"` |
| `uptime_seconds`        | number         | Service uptime in seconds                            |
| `processed_items_count` | integer        | Total number of items processed since startup        |
| `gpu_available`         | boolean        | Whether GPU is available (CUDA or MPS)               |
| `model_loaded`          | boolean        | Whether CLIP model is loaded and ready               |
| `model_name`            | string \| null | Name of the loaded CLIP model                        |
| `device`                | string \| null | Device being used (e.g., "cuda:0", "cpu", "mps")     |
| `request_queue_size`    | integer        | Current number of items in processing queue          |

---

### 2. Batch Embedding Generation

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

### Single Image from GCS

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

- **Concurrent Requests**: Limited by `request_queue_size`
- **Batch Size**: Maximum items per request limited by `MAX_BATCH_ITEMS`
- **VRAM**: Dynamic batching based on available GPU memory
- **GCS**: Limited by `GCS_MAX_WORKERS` concurrent downloads

---

## SDK Examples

### Python

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

### Typical Response Times

| Scenario                 | Items | Source | Response Time |
| ------------------------ | ----- | ------ | ------------- |
| Single image             | 1     | GCS    | 100-300ms     |
| Single image             | 1     | URL    | 500-1500ms    |
| Batch images             | 10    | GCS    | 800-1200ms    |
| Batch images             | 10    | URL    | 2000-5000ms   |
| Single video (20 frames) | 1     | GCS    | 1000-3000ms   |
| Single video (20 frames) | 1     | URL    | 3000-8000ms   |

### Memory Usage

- **Base model**: ~2GB VRAM (CLIP ViT-B/32)
- **Per image**: ~10MB VRAM
- **Per video frame**: ~10MB VRAM
- **Batch overhead**: ~100MB VRAM

---

_Last updated: August 5, 2025_
