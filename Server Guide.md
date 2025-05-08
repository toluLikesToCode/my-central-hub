# My Central Hub - Server Guide

This guide provides comprehensive documentation for the My Central Hub system architecture, focusing on server configuration, service communication, and advanced features.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Node.js Server Configuration](#nodejs-server-configuration)
3. [Python Embedding Service](#python-embedding-service)
4. [Communication Between Services](#communication-between-services)
5. [Advanced Features](#advanced-features)
6. [Error Handling](#error-handling)
7. [Environment Variables Reference](#environment-variables-reference)
8. [Configuration Quick Reference](#configuration-quick-reference)
9. [Troubleshooting](#troubleshooting)

## System Architecture

My Central Hub consists of two primary components:

1. **Node.js Server** - The main application server that handles HTTP requests, file operations, and coordinates services
2. **Python Embedding Service** - A FastAPI microservice that computes CLIP embeddings for images and videos

### Deployment Options

The system can be deployed in several ways:

- **Local Development**: Both services run directly on the host machine
- **Docker Containers**: Run in containers for improved isolation and consistency
- **Mixed Mode**: Node.js server on host with Python service in container

## Node.js Server Configuration

### Primary Configuration File

The main configuration is defined in `src/config/server.config.ts`, which controls:

```typescript
// Main configuration structure
{
  port: number,        // Server listen port
  publicDir: string,   // Public assets directory
  mediaDir: string,    // Media files directory
  adminKey: string,    // Admin authentication key
  logging: {...},      // Logging configuration
  features: {...},     // Feature toggles
  embedding: {...},    // Embedding service settings
  // Additional settings...
}
```

### Feature Toggles

Control which components are active:

```typescript
features: {
  metrics: true,         // Performance and usage metrics
  fileHosting: true,     // Static file serving
  fileStreaming: true,   // Media streaming
  embeddingService: true // CLIP embedding functionality
}
```

To disable any feature, set its value to `false`.

## Python Embedding Service

The Python embedding service is a FastAPI application that processes images and videos to generate CLIP embeddings.

### Core Components

1. **`server.py`** - The FastAPI entry point and HTTP interface
2. **`embedding_service_helper.py`** - The embedding computation engine

### Service Architecture

The embedding service uses a layered architecture:

```
┌─────────────────────────────────────┐
│ FastAPI HTTP Interface (server.py)  │
├─────────────────────────────────────┤
│ Embedding Processing Pipeline       │
│ ┌───────────────┐ ┌───────────────┐ │
│ │ CLIPEmbedder  │ │VideoProcessor │ │
│ └───────────────┘ └───────────────┘ │
├─────────────────────────────────────┤
│ Hardware Abstraction Layer          │
│ (CUDA, MPS, CPU, FFmpeg Accel)      │
└─────────────────────────────────────┘
```

### Hardware Acceleration

The embedding service automatically detects and uses available hardware acceleration:

1. **CUDA** - For NVIDIA GPUs on Linux/Windows
2. **MPS** - Metal Performance Shaders for macOS
3. **VideoToolbox** - Hardware-accelerated video decoding on macOS
4. **CPU with Optimized Libraries** - Fallback for all platforms

### Optimized Video Processing

The VideoProcessor component employs several optimizations for efficient frame extraction:

1. **Intelligent Frame Seeking**:

   - Uses FFmpeg's keyframe-based seeking for faster access
   - Places the `-ss` option after the input source for more efficient seeking

2. **Hardware-Accelerated Decoding**:

   - Automatically detects and uses GPU-accelerated video decoding
   - Supports CUDA, VideoToolbox and other hardware decoders

3. **Raw Format Processing**:

   - Uses uncompressed RGB24 format to eliminate encoding/decoding overhead
   - Provides better image quality for embedding processing

4. **Adaptive Resource Management**:

   - Limits thread usage to prevent CPU resource abuse
   - Implements dynamic timeouts based on video duration and complexity
   - Controls buffer sizes for memory efficiency

5. **Robust Error Handling**:
   - Implements a multi-tiered fallback system for maximum reliability
   - Provides format-specific processing for problematic videos
   - Comprehensive error reporting with context

### API Endpoints

- **`GET /health`** - Service health check
- **`GET /gpu-metrics`** - GPU utilization statistics (when available)
- **`POST /embed`** - Compute embeddings for uploaded media files

### Configuration

The Python service uses these key environment variables:

- `PORT` - Service port (default: 3456)
- `CLIP_MODEL` - Model name (default: "openai/clip-vit-base-patch32")

## Communication Between Services

The Node.js server connects to the Python embedding service using HTTP requests. This communication is managed by `EmbeddingHttpClient` which implements:

1. **Dynamic timeouts** based on file size and type
2. **Automatic retries** with exponential backoff
3. **Categorized error handling** for different failure types
4. **Consistent metadata transformation** between snake_case and camelCase

### Connection Configuration

```typescript
embedding: {
  serviceUrl: 'http://localhost:3456', // Python service URL
  maxRetries: 3,                       // Maximum retry attempts
  retryDelayMs: 1000,                  // Base delay between retries
  timeoutMs: 300000,                   // Base timeout (5 minutes)
  // Additional timeout settings...
}
```

When running in Docker, ensure `EMBEDDING_SERVICE_URL` is set to the container service name (e.g., `http://embedding-service:3456`).

## Advanced Features

### Dynamic Timeout Calculation

The system intelligently calculates request timeouts based on:

- File size (1 second per MB)
- Media type (videos get longer timeouts)
- Configurable minimum and maximum bounds

```typescript
// Default timeout configuration
{
  baseTimeoutMs: 300000,        // 5 minutes default
  minTimeoutMs: 30000,          // 30 seconds minimum
  maxTimeoutMs: 1800000,        // 30 minutes maximum
  videoMultiplier: 4,           // Videos get 4x the timeout of images
  timeoutPerMbMs: 1000          // 1 second per MB of file size
}
```

### Content-Aware Video Processing

The Python service uses advanced techniques for extracting the most representative frames from videos:

1. **Scene Detection**:

   - Identifies significant scene changes in the video
   - Extracts frames from the middle of each detected scene
   - Ensures varied content representation

2. **Visual Entropy Analysis**:

   - Calculates the information richness of potential frames
   - Prioritizes visually distinctive frames with higher entropy
   - Skip low-information frames (black screens, solid colors)

3. **Temporal Diversity Enforcement**:

   - Maintains minimum time gaps between selected frames
   - Ensures coverage across the entire video duration
   - Intelligently selects keyframes for optimal representation

4. **Batch Processing**:
   - Processes multiple files in a single request
   - Uses thread pooling for parallel execution
   - Caching to avoid redundant computation

### Metadata Propagation

The system preserves rich metadata from the Python embedding process:

- Model information (name, device)
- Processing parameters (augmentation, frame count)
- Video analysis data (scene detection, entropy values)
- Performance metrics (processing time)

All metadata is transformed to camelCase format and validated against the JSON schema.

### Schema Validation

All embedding results are validated against the `clipCache.schema.json` schema to ensure:

- Data integrity
- Consistent structure
- Compatibility with client applications

## Error Handling

The system implements comprehensive error handling to distinguish between:

### 1. Network Errors

Issues connecting to the Python service:

- Connection refused (service down)
- Timeouts (service overloaded or unresponsive)
- Network unreachable

### 2. Service Errors

Valid HTTP responses with error status codes:

- 400-level errors (client errors)
- 500-level errors (server errors)

### 3. Processing Errors

Issues during embedding computation:

- File format problems
- Resource limitations (GPU memory)
- Model loading failures

Each error type is clearly identified in log messages and ClipCacheEntry error fields to help with troubleshooting.

## Environment Variables Reference

### Node.js Server

| Variable              | Default                 | Description                  |
| --------------------- | ----------------------- | ---------------------------- |
| PORT                  | 8080                    | Server listen port           |
| PUBLIC_DIR            | './public'              | Public assets directory      |
| MEDIA_DIR             | './public/media'        | Media files directory        |
| LOG_LEVEL             | 'info'                  | Logging level                |
| EMBEDDING_SERVICE_URL | 'http://localhost:3456' | URL to the embedding service |
| EMBEDDING_TIMEOUT     | '300000'                | Base timeout in milliseconds |

### Python Embedding Service

| Variable            | Default                        | Description                |
| ------------------- | ------------------------------ | -------------------------- |
| PORT                | 3456                           | Service listen port        |
| CLIP_MODEL          | 'openai/clip-vit-base-patch32' | CLIP model name            |
| LOG_LEVEL           | 'info'                         | Logging level              |
| NUM_FRAMES          | '20'                           | Number of frames for video |
| ENABLE_AUGMENTATION | 'false'                        | Enable data augmentation   |
| MAX_THREADS         | '8'                            | Maximum worker threads     |

## Configuration Quick Reference

### Modifying Configuration (Recommended Method)

Use environment variables in a `.env` file:

```
# Node.js server
PORT=9000
LOG_LEVEL=debug
EMBEDDING_SERVICE_URL=http://embedding-service:3456

# Python service
CLIP_MODEL=openai/clip-vit-large-patch14
```

### Docker Configuration

For Docker Compose:

```yaml
services:
  main-server:
    image: my-central-hub
    environment:
      - PORT=9000
      - EMBEDDING_SERVICE_URL=http://embedding-service:3456

  embedding-service:
    image: embedding-service
    environment:
      - PORT=3456
      - CLIP_MODEL=openai/clip-vit-large-patch14
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Troubleshooting

### Common Issues

#### 1. Embedding Service Connection Errors

If the Node.js server cannot connect to the embedding service:

- Verify the embedding service is running
- Check the `EMBEDDING_SERVICE_URL` is correct
- For Docker: Ensure both services are on the same network
- Check logs for specific error types (network vs. service)

#### 2. Timeout Errors

If requests time out:

- Increase `EMBEDDING_TIMEOUT` for large files
- Adjust timeout multipliers for videos
- Check Python service logs for processing bottlenecks

#### 3. Metadata Validation Errors

If schema validation fails:

- Check the error logs for specific validation issues
- Verify clients can handle the metadata format

#### 4. Video Processing Errors

If video processing fails:

- Check the `logs/python_embeddings.log` for FFmpeg error details
- Verify FFmpeg is installed and in the PATH
- For corrupted videos, try preprocessing them with `ffmpeg -i input.mp4 -c copy fixed.mp4`

### Diagnostic Tools

#### Health Check

Verify the embedding service is running:

```
GET http://localhost:3456/health
```

#### GPU Metrics

Check GPU usage (if available):

```
GET http://localhost:3456/metrics/gpu
```

#### Log Files

Check the relevant log files:

- `logs/embeddings.log` - Node.js embedding client logs
- `logs/python_embeddings.log` - Python service logs
- `logs/embedding_validation.log` - Schema validation issues
