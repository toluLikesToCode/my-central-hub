# My Central Hub - Client API Reference

This document provides up-to-date information for client-side developers integrating with the My Central Hub embedding service API, reflecting the latest request/response formats, error handling, and metadata.

## API Endpoints

### 1. Embedding Endpoint

```
POST /api/embeddings
```

**Parameters (JSON body):**

- `files`: Array of file paths or URLs to process (preferred, documented)
- `imagePaths`: (legacy) Also accepted for backward compatibility

**Example:**

```javascript
fetch('/api/embeddings', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    files: ['/media/image1.jpg', '/media/video1.mp4'],
    // or: imagePaths: ['/media/image1.jpg', '/media/video1.mp4']
  }),
});
```

**Request Notes:**

- You may provide either `files` or `imagePaths` (array of strings). `files` is preferred.
- Single string values are also accepted and will be converted to an array.
- Path traversal attempts (e.g., "../file.jpg") are automatically sanitized.
- All paths are validated to ensure they're within the allowed media directory.

**Response:**
The response is a JSON object with file paths as keys and embedding results as values.

```typescript
{
  "/media/image1.jpg": ClipCacheEntry,
  "/media/video1.mp4": ClipCacheEntry
}
```

### 2. Status Endpoint

```
GET /api/embeddings/status
```

Use this endpoint to check the status of the embedding service before submitting large batches. It can help determine if the service is ready to accept requests or might be experiencing issues.

**Example:**

```javascript
fetch('/api/embeddings/status')
  .then((response) => response.json())
  .then((status) => {
    if (status.state === 'IDLE') {
      // Service ready for processing
      submitEmbeddingRequest();
    } else if (status.state === 'ERROR') {
      // Service has issues
      showErrorMessage(`Service error: ${status.lastError}`);
    } else if (status.state === 'PROCESSING') {
      // Service is busy
      showBusyMessage(`Currently processing ${status.currentBatch?.current}`);
    }
  });
```

**Response:**

```typescript
{
  state: 'IDLE' | 'PROCESSING' | 'ERROR' | 'STARTING' | 'STOPPED',
  pid: number | null,
  isStarting: boolean,
  isProcessing: boolean,
  queueLength: number,
  currentBatch?: {
    count: number,
    total: number,
    current: string
  },
  lastError?: string
}
```

### 3. Shutdown Endpoint (DEPRECATED)

```
POST /api/embeddings/shutdown
```

> ⚠️ **Deprecated**: This endpoint will be removed in the next major version. Use system management tools to control the service instead.

## ClipCacheEntry Structure

The `ClipCacheEntry` object follows this structure:

```typescript
interface ClipCacheEntry {
  schemaVersion: string; // "1.0.0"
  filePath: string; // Original file path
  mediaType: 'image' | 'video';
  mtime: number; // Modification timestamp
  fileSize: number; // File size in bytes
  dimensions: { width: number; height: number };
  embedding: number[]; // The embedding vector (empty if failed)
  embeddingModel: string; // e.g. "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
  embeddingConfig: {
    augmentation?: boolean;
    numFrames?: number | null;
    samplingMethod?: string;
    deviceType?: string;
    batchId?: string;
    [key: string]: any;
  };
  processingTimestamp: string; // ISO timestamp
  duration?: number | null; // Video duration in seconds (null for images)
  debugMetadata?: { 
    model: string;
    enableAugmentation: boolean;
    processingTimeMs: number;
    device: string;
    methodUsed?: string;
    calculatedTimeout?: number;
    clientProcessingTime?: number;
    errorType?: string;
    [key: string]: any;
  };
  error?: string; // Error message (if processing failed)
  detail?: string; // Detailed error information
}
```

## Error Handling

**HTTP Status Codes:**

| Status Code | Meaning                                      |
| ----------- | -------------------------------------------- |
| 200         | Success                                      |
| 400         | Bad Request (invalid input)                  |
| 404         | File Not Found                               |
| 405         | Method Not Allowed (use POST)                |
| 429         | Too Many Requests                            |
| 500         | Server Error                                 |
| 503         | Service Unavailable (embedding service down) |

**Error Fields in Response:**

- `error`: Short error message
- `detail`: Detailed error information
- `debugMetadata.errorType`: One of `"network_error"`, `"service_error"`, `"processing_error"`, `"timeout_error"`

**Example Error Response:**

```json
{
  "/media/image1.jpg": {
    "schemaVersion": "1.0.0",
    "filePath": "/media/image1.jpg",
    "mediaType": "image",
    "mtime": 0,
    "fileSize": 0,
    "dimensions": { "width": 1, "height": 1 },
    "embedding": [],
    "embeddingModel": "unknown",
    "embeddingConfig": {},
    "processingTimestamp": "2024-06-01T12:00:00Z",
    "error": "Service error: The embedding service responded with an error (500)",
    "detail": "Detailed error information",
    "debugMetadata": {
      "errorType": "service_error",
      "statusCode": 500,
      "errorTime": "2024-06-01T12:00:00Z"
    }
  }
}
```

## Debug Metadata Fields

The `debugMetadata` field may include:

```typescript
debugMetadata: {
  model: string,                   // The exact model used
  enableAugmentation: boolean,     // Whether augmentation was applied
  processingTimeMs: number,        // Processing time in milliseconds
  device: "cuda" | "mps" | "cpu",  // The device used for inference
  numFrames?: number,              // Number of frames processed (for videos)
  methodUsed?: string,             // Frame sampling method (for videos)
  sceneCount?: number,             // Number of scenes detected (for videos)
  selectedTimes?: number[],        // Timestamps of chosen frames (for videos)
  entropyValues?: [number, number][], // Timestamp/entropy pairs for frame selection
  calculatedTimeout?: number,      // Timeout used for the request
  clientProcessingTime?: number,   // Total processing time on client side
  batchId?: string,                // Batch processing identifier
  errorType?: string,              // For failed requests: "network_error", "service_error", "processing_error", "timeout_error"
  statusCode?: number,             // HTTP status code for service errors
  errorTime?: string               // Time when error occurred
}
```

## Best Practices

- **Use `files` as the request key** for new clients; `imagePaths` is still supported for backward compatibility.
- **Always check for errors** in the response before using the embedding.
- **Check service status** before submitting large batches.
- **Process in batches** for large collections.
- **Handle different error types** appropriately (see error handling above).
- **For large videos**, consider pre-processing or splitting for best performance.
- **Ensure file paths are relative** to the media directory and don't contain path traversal attempts.

## Embedding Vector

The `embedding` field contains a floating-point array representing the CLIP embedding:

- Standard dimension: 512 for base models, 768 for large models
- Values are normalized (unit vector)
- Empty array (`[]`) indicates embedding generation failed

## Client Implementation Best Practices

1. **Check service status before submitting large batches**:

   ```javascript
   async function processWithStatusCheck(files) {
     // Check service status first
     const status = await fetch('/api/embeddings/status').then((r) => r.json());

     if (status.state === 'ERROR') {
       console.error('Service error:', status.lastError);
       return { error: status.lastError };
     }

     if (status.state === 'PROCESSING' && files.length > 10) {
       console.warn('Service busy, try again later');
       return { error: 'Service busy' };
     }

     // Proceed with embedding request
     return fetch('/api/embeddings', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({ files }),
     }).then((r) => r.json());
   }
   ```

2. **Always check for errors** before using the embedding:

   ```javascript
   if (result.error) {
     console.error(`Error processing ${result.filePath}: ${result.error}`);
     // Handle error appropriately
   } else {
     // Process valid embedding
   }
   ```

3. **Handle different error types** differently:

   ```javascript
   if (result.debugMetadata?.errorType === 'network_error') {
     // Show connectivity troubleshooting
   } else if (result.debugMetadata?.errorType === 'timeout_error') {
     // Suggest trying with smaller file
   }
   ```

4. **Consider file size limitations** for timeout-sensitive operations:

   - Images: Generally process quickly
   - Short videos (< 1 minute): Medium processing time
   - Long videos (1-5 minutes): Longer processing time
   - Very large videos (> 5 minutes or > 100MB): May require specialized handling (see "Handling Large Videos" section)

5. **Process in batches** for large collections:
   ```javascript
   async function processBatches(allFiles, batchSize = 10) {
     const results = {};
     for (let i = 0; i < allFiles.length; i += batchSize) {
       const batch = allFiles.slice(i, i + batchSize);
       const batchResults = await getEmbeddingWithRetry(batch);
       Object.assign(results, batchResults);
     }
     return results;
   }
   ```

## Handling Large Videos (NEW)

The embedding service has been enhanced to better handle large video files through:

1. **Dynamic Timeouts**: Timeouts are calculated based on file size, media type, and video duration, with exponential backoff on retries.

2. **Intelligent Frame Selection**: Videos are processed using a combination of:
   - Scene detection for content-aware frame selection
   - Visual entropy calculation to identify salient frames
   - Temporal diversity enforcement to ensure varied frame selection

3. **Advanced Metadata**: Rich debug information about video processing:
   - Frame sampling method used
   - Number of frames processed
   - Scene detection statistics
   - Selected frame timestamps

4. **Error Resilience**: Multiple fallback strategies if initial methods fail:
   - Fallback from hardware acceleration to software processing
   - Fallback from batch processing to frame-by-frame extraction
   - Detailed error reporting for debugging

### Best Practices for Large Videos

When working with large video files:

1. **Pre-process videos** when possible:

   - Consider reducing video resolution or bitrate for faster processing
   - Split very long videos into smaller segments (1-2 minute chunks)
   - Extract key frames or scenes as still images

2. **Set reasonable expectations** for processing time:

   - Videos up to 100MB: Usually process within standard timeouts
   - Videos 100MB-500MB: May require extended time
   - Videos over 500MB: Consider pre-processing into smaller segments

3. **Implement client-side progress indicators** for user feedback:

   - Use the status endpoint to poll for processing status
   - Display estimated processing time based on file size/duration

4. **Handle timeout errors gracefully**:

   ```javascript
   if (result.error && result.debugMetadata?.errorType === 'timeout_error') {
     // Suggest pre-processing the video
     showMessage('This video is too large. Consider using a shorter clip or lower resolution.');
   }
   ```

5. **Utilize the rich metadata** for better user feedback:
   ```javascript
   // Show processing details
   if (result.debugMetadata?.methodUsed === 'scene_detection') {
     console.info(`Video was analyzed with scene detection, found ${result.debugMetadata.sceneCount} scenes`);
   }
   ```
