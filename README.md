# My Central Hub

My Central Hub is a robust and scalable platform designed to manage and serve media files efficiently. It provides APIs for streaming, listing, and handling media files, making it ideal for applications requiring high-performance file management.

## Features

- **Media Streaming**: Stream media files with support for HTTP range requests.
- **File Management**: List and manage media files stored in the server.
- **Error Handling**: Comprehensive error handling for invalid requests and missing files.
- **Extensible Architecture**: Modular design for easy integration and extension.

## Project Structure

```plaintext
compress-central-hub.sh
eslint.config.js
jest.config.js
Makefile
my-central-hub.xml
package.json
README.md
src/
  main.ts
  config/
    server.config.ts
  core/
    httpParser.ts
    parser.ts
    router.ts
    server.ts
  entities/
    http.ts
    sendResponse.ts
  modules/
    file-streamer/
      fileService.ts
      fileStreamingController.ts
      index.ts
  routes/
    files.routes.ts
    index.ts
    stream.routes.ts
  utils/
    helpers.ts
    httpHelpers.ts
    logger.ts
    mimeTypes.ts
stress/
tests/
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/toluLikesToCode/my-central-hub.git
   ```

2. Navigate to the project directory:

   ```bash
   cd my-central-hub
   ```

3. Install dependencies:

   ```bash
   npm install
   ```

## Usage

### Development Server

Start the development server with:

```bash
npm run dev
```

### Build for Production

Build the project for production with:

```bash
npm run build
```

### Run Tests

Run all tests:

```bash
npm test
```

Run CI-specific tests (excluding stress tests):

```bash
npm run test:ci
```

## Configuration

The project uses environment variables for configuration. Create a `.env` file in the root directory and specify the following variables:

```bash
PORT=8080
PUBLIC_DIR=public
MEDIA_DIR=media
HEADER_TIMEOUT_MS=5000
BODY_TIMEOUT_MS=10000
```

## About This Project

This project, My Central Hub, is a personal endeavor to deepen my understanding of networking and server-side development. It serves as a learning platform to explore concepts such as HTTP protocols, media streaming, and efficient file handling.

### Future Plans

I plan to continue extending the features of this project to transform it into a full-fledged personal home server. Some of the planned enhancements include:

- **User Authentication**: Adding secure user authentication and authorization mechanisms.
- **Media Library Management**: Implementing features to organize and categorize media files.
- **Remote Access**: Enabling secure remote access to the server from anywhere.
- **Backup and Sync**: Adding functionality to back up and sync files across devices.
- **Enhanced Streaming**: Supporting adaptive streaming for better performance on varying network conditions.
- **AI Model Hosting**: Hosting AI models for inference and experimentation.
- **Web Hosting**: Serving static and dynamic web applications.
- **Database Management**: Managing databases for storing and querying structured data.

This project is a continuous work in progress, and I am excited to see how it evolves over time.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear and concise messages.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or support, please open an issue on the [GitHub repository](https://github.com/toluLikesToCode/my-central-hub/issues).
