# My Central Hub

My Central Hub is a robust, modular platform for managing and serving media files, with APIs for streaming, file management, application metrics, and advanced AI-powered embedding via a dedicated microservice. It is designed for extensibility, performance, and learning about server-side networking and service orchestration.

## Architecture Overview

- **Node.js Server**: Handles HTTP requests, file operations, metrics, and coordinates with the embedding service.
- **Python Embedding Service (FastAPI)**: Computes CLIP embeddings for images and videos via an HTTP API. Now fully decoupled from the Node.js process.
- **Docker Compose Orchestration**: Both services can be run together in containers, sharing a network and media volume for seamless integration.

## Features

- **Media Streaming**: Stream video and audio files with HTTP range support.
- **File Management**: List, upload, download, and delete files via RESTful APIs.
- **Metrics Collection**: Modular endpoints for collecting app-specific metrics (e.g., gallery-generator).
- **AI Embedding Microservice**: Upload images/videos to the Python FastAPI service for CLIP embeddings, with advanced error handling and retry logic.
- **Extensible Architecture**: Add new modules and endpoints easily.
- **Comprehensive Error Handling**: Consistent error responses for API and non-API routes, with categorized error types for embedding.
- **Configurable**: Environment-based configuration and feature toggling.
- **Advanced Video Processing**: FFmpeg optimizations for frame extraction, hardware acceleration, and adaptive resource management (in progress).

## Project Structure

```
src/
  config/             # Server and feature configuration
  core/               # HTTP parser, router, server, middlewares
  entities/           # HTTP request/response types, helpers
  modules/
    app-metrics/      # Metrics endpoints (per-app)
    embeddings/       # Embedding logic and API integration
    file-hosting/     # File management (list, upload, download, delete)
    file-streaming/   # Streaming controller (deprecated, see file-hosting)
  routes/             # Route registration (RESTful API)
    embeddings.routes.ts   # Embeddings API routes
    file-hosting.routes.ts # File hosting API routes
    files.routes.ts        # File listing/filtering API (deprecated, see file-hosting)
    metrics.routes.ts      # Metrics API routes
    stream.routes.ts       # Streaming API routes
    index.ts               # Route aggregator
  utils/              # Helpers, logger, mime types, date formatting
services/
  embedding-service/  # Python FastAPI CLIP embedding microservice
schemas/              # JSON schema definitions for validation
public/
  media/              # Uploaded media files (images, videos, gifs, audio, etc.)
data/                 # SQLite databases (metrics, file stats, etc.)
stress/               # Stress tests
logs/                 # Application and service logs
build-logs/           # Build and deployment logs
tests/                # Unit, integration, and end-to-end tests
```

- Deprecated modules are marked as such. See `modules/file-streaming/`.
- Embedding microservice is in `services/embedding-service/` (Python).
- All API routes are in `src/routes/`.
- Media files are stored in `public/media/`.
- Database files are in `data/`.
- JSON schemas for validation are in `schemas/`.

## Embedding Microservice (CLIP)

- The Node.js server communicates with the Python embedding service via HTTP POST requests using FormData for file uploads.
- The `EmbeddingHttpClient` handles retries, backoff, and error categorization.
- Embedding results include rich metadata, validated against a JSON schema.
- All configuration is now via HTTP client settings (see below).

### Running Both Services (Docker Compose)

A `docker-compose.yml` is provided:

- `api`: Node.js server
- `embed`: Python FastAPI embedding service
- Shared network and media volume
- Health checks and resource limits included

To start both services:

```sh
docker-compose up --build
```

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/toluLikesToCode/my-central-hub.git
   cd my-central-hub
   ```
2. Install dependencies:
   ```sh
   npm install
   ```

## Usage

### Development Server

```sh
npm run dev
```

### Build for Production

```sh
npm run build
npm start
```

### Run Tests

```sh
npm test         # All tests
npm run test:ci  # Excludes stress tests
npm run test:stress # Stress tests only
```

## Configuration

Create a .env file in the project root to override defaults:

```
PORT=8080
PUBLIC_DIR=public
MEDIA_DIR=media
HEADER_TIMEOUT_MS=5000
BODY_TIMEOUT_MS=10000
DB_PATH=...
EMBEDDING_SERVICE_URL=...
```

Feature toggles are in server.config.ts:

```typescript
features: {
  metrics: true,
  fileHosting: true,
  fileStreaming: true,
  embedding: true,
}
```

Advanced timeout, retry, and resource settings for embedding are available (see `Server Guide.md`).

## RESTful API Conventions

- All API endpoints are under `/api/`.
- Use plural nouns for collections (e.g., `/api/files`).
- Use route parameters for single resources (e.g., `/api/files/:filename`).
- Metrics endpoints: `/api/metrics/:app`.

## Example Endpoints & Usage

- **List Files**:  
  `GET /api/files`
- **Upload File**:  
  `POST /api/files` (multipart/form-data)
- **Download File**:  
  `GET /api/files/:filename`
- **Delete File**:  
  `DELETE /api/files/:filename`
- **Stream File**:  
  `GET /api/stream?file=filename.mp4`
- **Submit Metrics**:  
  `POST /api/metrics/gallery-generator` (JSON body)
- **Generate Embedding**:  
  `POST /api/embed` (multipart/form-data)

Example using curl:

```sh
curl http://localhost:8080/api/files
curl -F 'file=@/path/to/file.jpg' http://localhost:8080/api/files
curl http://localhost:8080/api/files/filename.jpg -o filename.jpg
curl -X DELETE http://localhost:8080/api/files/filename.jpg
curl -X POST http://localhost:8080/api/metrics/gallery-generator -H 'Content-Type: application/json' -d '{"event":"view","user":"alice"}'
curl -F 'file=@/path/to/video.mp4' http://localhost:8080/api/embed
```

## Modular Architecture & Extending the Project

- Add new modules under modules.
- Register routes in routes.
- Use feature toggles in config to enable/disable modules.
- For new metrics endpoints, see README.md.

## Testing

- Unit tests mock the HTTP embedding client (no more Python process mocks).
- Integration tests verify HTTP communication between Node.js and Python services.
- Stress tests simulate concurrent embedding requests.

## Advanced Features

- **Dynamic Timeout Calculation**: Based on file size/type, with min/max bounds.
- **Content-Aware Video Processing**: Scene detection, entropy analysis, and temporal diversity for frame selection.
- **Hardware Acceleration**: CUDA, VideoToolbox, and CPU fallback for FFmpeg.
- **Robust Error Handling**: Network, service, and processing errors are clearly categorized and logged.

## Migration & Cleanup

- All legacy Python process code and config flags have been removed.
- Documentation and codebase reflect the new HTTP-based architecture.

## More About This Project

This project is a personal learning platform for networking, HTTP, and server-side development. Planned enhancements include:

- User authentication
- Media library management
- Remote access
- Backup and sync
- Adaptive streaming
- AI model hosting
- Web hosting
- Database management

## In Progress

- Finalizing FFmpeg optimizations for video embedding (see `embed-microservice-plan.md` and `Server Guide.md`).
- Updating benchmark metrics and adding pipeline diagrams.

## Contributing

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Submit a pull request.

## License

MIT License. See LICENSE file.

## Contact

For questions or support, open an issue on [GitHub](https://github.com/toluLikesToCode/my-central-hub/issues).
