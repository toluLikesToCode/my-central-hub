Here’s an updated main README for your project, reflecting the current codebase, features, and usage:

---

# My Central Hub

My Central Hub is a robust, modular platform for managing and serving media files, with APIs for streaming, file management, and application metrics. It is designed for extensibility, performance, and learning about server-side networking.

## Features

- **Media Streaming**: Stream video and audio files with HTTP range support.
- **File Management**: List, upload, download, and delete files via RESTful APIs.
- **Metrics Collection**: Modular endpoints for collecting app-specific metrics (e.g., gallery-generator).
- **Extensible Architecture**: Add new modules and endpoints easily.
- **Comprehensive Error Handling**: Consistent error responses for API and non-API routes.
- **Configurable**: Environment-based configuration and feature toggling.

## Project Structure

```
src/
  config/           # Server and feature configuration
  core/             # HTTP parser, router, server
  entities/         # HTTP request/response types
  modules/
    app-metrics/    # Metrics endpoints (per-app)
    file-hosting/   # File management (list, upload, download, delete)
    file-streaming/ # Streaming controller (deprecated, see file-hosting)
  routes/           # Route registration (RESTful API)
  utils/            # Helpers, logger, mime types
tests/              # Unit and integration tests
stress/             # Stress tests
public/             # Static/public files
data/               # SQLite database (metrics)
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
DB_PATH=data/metrics.db
```

Feature toggles are in server.config.ts:

```typescript
features: {
  metrics: true,
  fileHosting: true,
  fileStreaming: true,
}
```

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

Example using curl:
```sh
curl http://localhost:8080/api/files
curl -F 'file=@/path/to/file.jpg' http://localhost:8080/api/files
curl http://localhost:8080/api/files/filename.jpg -o filename.jpg
curl -X DELETE http://localhost:8080/api/files/filename.jpg
curl -X POST http://localhost:8080/api/metrics/gallery-generator -H 'Content-Type: application/json' -d '{"event":"view","user":"alice"}'
```

## Modular Architecture & Extending the Project

- Add new modules under modules.
- Register routes in routes.
- Use feature toggles in config to enable/disable modules.
- For new metrics endpoints, see README.md.

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

## Contributing

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Submit a pull request.

## License

MIT License. See LICENSE file.

## Contact

For questions or support, open an issue on [GitHub](https://github.com/toluLikesToCode/my-central-hub/issues).

---

Let me know if you want this written to your README.md file or need further customization!
