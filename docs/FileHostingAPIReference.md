# File Hosting Service API Reference

## Overview

The File Hosting Service provides a comprehensive API for managing files within the system. It includes capabilities for file retrieval, listing, uploading, deletion, and directory management with advanced features like pagination, filtering, partial content (range requests), and compression.

This document provides detailed information about each API endpoint, including request parameters, response formats, status codes, and examples.

## Base URL

All API endpoints are relative to the base URL of your server installation.

```
http://<your-server>:<port>/api
```

## Authentication

Most endpoints do not require authentication. The `/api/files/cache` endpoint requires an admin key specified as an `x-admin-key` header.

## Response Format

All API responses use standard HTTP status codes. Most endpoints return JSON-formatted responses with the following general structure:

```json
{
  "success": true,
  "message": "Operation completed successfully"
  // Additional response data varies by endpoint
}
```

Error responses follow a similar structure:

```json
{
  "success": false,
  "message": "Error message describing what went wrong"
  // Additional error details may be included
}
```

## Common Status Codes

| Status Code | Description                                                           |
| ----------- | --------------------------------------------------------------------- |
| 200         | Success - The request was successful                                  |
| 201         | Created - A resource was successfully created                         |
| 304         | Not Modified - The resource hasn't changed (for conditional requests) |
| 400         | Bad Request - The request was malformed or invalid                    |
| 404         | Not Found - The requested resource doesn't exist                      |
| 416         | Range Not Satisfiable - The requested range is not valid              |
| 500         | Internal Server Error - Something went wrong on the server            |

---

## File Operations

### Get a File

Retrieves a file with support for range requests, compression, and conditional requests.

#### Endpoint

```
GET /files/:filename
GET /files?file=<filename>
```

#### URL Parameters

| Parameter | Description                      |
| --------- | -------------------------------- |
| filename  | The name of the file to retrieve |

#### Query Parameters

| Parameter | Required                     | Description              |
| --------- | ---------------------------- | ------------------------ |
| file      | Yes (if path param not used) | The filename to retrieve |

#### Request Headers

| Header            | Required | Description                                              |
| ----------------- | -------- | -------------------------------------------------------- |
| Range             | No       | Request only part of the file. Format: `bytes=start-end` |
| If-None-Match     | No       | ETag for conditional request                             |
| If-Modified-Since | No       | Date for conditional request                             |
| Accept-Encoding   | No       | Supported encodings, e.g., `gzip, deflate, br`           |

#### Response

##### Success Response (200 OK)

Full file content with appropriate Content-Type headers.

**Headers:**

```
Content-Type: <mime-type-of-file>
Content-Length: <size-in-bytes>
Accept-Ranges: bytes
Cache-Control: public, max-age=86400, stale-while-revalidate=43200
ETag: "<size>-<modified-timestamp>"
Last-Modified: <modified-date>
X-Content-Type-Options: nosniff
```

For compressible content types when the client supports compression:

```
Content-Encoding: <compression-type>
Vary: Accept-Encoding
```

##### Partial Content (206)

**Headers:**

```
Content-Type: <mime-type-of-file>
Content-Length: <size-of-range>
Content-Range: bytes <start>-<end>/<total-size>
Accept-Ranges: bytes
Cache-Control: public, max-age=86400, stale-while-revalidate=43200
ETag: "<size>-<modified-timestamp>"
Last-Modified: <modified-date>
```

##### Not Modified (304)

Returned when the file hasn't changed since the specified If-Modified-Since date or matches the provided ETag.

**Headers:**

```
ETag: "<size>-<modified-timestamp>"
Last-Modified: <modified-date>
Cache-Control: public, max-age=86400
```

##### Error Responses

| Status Code | Description                                            |
| ----------- | ------------------------------------------------------ |
| 400         | Missing required "file" parameter                      |
| 404         | File not found or inaccessible                         |
| 416         | Range Not Satisfiable - The requested range is invalid |
| 500         | Error creating file stream or other server error       |

#### Examples

**Basic File Request:**

```
GET /api/files/example.jpg
```

**Range Request (First 1000 bytes):**

```
GET /api/files/example.jpg
Range: bytes=0-999
```

**Conditional Request:**

```
GET /api/files/example.jpg
If-None-Match: "12345-1620000000000"
```

---

### List Files

Returns a paginated, sortable, and filterable list of available files.

#### Endpoint

```
GET /files
```

#### Query Parameters

| Parameter | Required | Default | Description                                 |
| --------- | -------- | ------- | ------------------------------------------- |
| page      | No       | 1       | Page number for pagination                  |
| limit     | No       | 20      | Number of items per page                    |
| sort      | No       | "name"  | Sort field (name, size, date, type)         |
| order     | No       | "asc"   | Sort order (asc, desc)                      |
| type      | No       | -       | Filter by MIME type prefix (e.g., "image/") |
| search    | No       | -       | Search term for filename or MIME type       |
| dateFrom  | No       | -       | Start date for filtering (ISO format)       |
| dateTo    | No       | -       | End date for filtering (ISO format)         |
| sizeFrom  | No       | 0       | Minimum file size in bytes                  |
| sizeTo    | No       | 0       | Maximum file size in bytes (0 = no limit)   |

#### Response

##### Success Response (200 OK)

```json
{
  "files": [
    {
      "name": "example.jpg",
      "path": "example.jpg",
      "size": 12345,
      "mimeType": "image/jpeg",
      "lastModified": "2023-05-10T15:30:45Z",
      "url": "/api/files/example.jpg"
    }
    // Additional files...
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "totalFiles": 100,
    "totalPages": 5,
    "hasNextPage": true,
    "hasPrevPage": false
  },
  "sorting": {
    "sort": "name",
    "order": "asc"
  },
  "filter": {
    "type": "image/",
    "search": "example",
    "dateRange": {
      "from": "2023-01-01",
      "to": "2023-05-10"
    },
    "sizeRange": {
      "from": 0,
      "to": 0
    }
  },
  "_links": {
    "self": "/api/files?page=1&limit=20&sort=name&order=asc",
    "first": "/api/files?page=1&limit=20&sort=name&order=asc",
    "last": "/api/files?page=5&limit=20&sort=name&order=asc",
    "next": "/api/files?page=2&limit=20&sort=name&order=asc",
    "prev": null
  }
}
```

##### Error Response (500 Internal Server Error)

```json
{
  "error": "Failed to list files",
  "message": "Error details",
  "timestamp": "2023-05-10T15:30:45Z"
}
```

#### Examples

**Basic Listing:**

```
GET /api/files
```

**Paginated Image Files:**

```
GET /api/files?page=2&limit=10&type=image/
```

**Search with Sorting:**

```
GET /api/files?search=report&sort=date&order=desc
```

---

### Search Files

Advanced file search with filtering.

#### Endpoint

```
GET /files/search
```

#### Query Parameters

Same as the List Files endpoint.

#### Response

Same as the List Files endpoint.

---

### Upload a File

Uploads a new file to the server.

#### Endpoint

```
POST /files
```

#### Query Parameters

| Parameter | Required | Description                                 |
| --------- | -------- | ------------------------------------------- |
| file      | No       | Filename to save as (alternative to header) |

#### Request Headers

| Header         | Required | Description                                      |
| -------------- | -------- | ------------------------------------------------ |
| Content-Type   | Yes      | MIME type of the file being uploaded             |
| Content-Length | Yes      | Size of the file in bytes                        |
| x-filename     | No       | Filename to save as (alternative to query param) |

#### Request Body

The raw file content as binary data.

#### Response

##### Success Response (200 OK)

```json
{
  "success": true,
  "fileName": "example.jpg",
  "size": 12345,
  "mimeType": "image/jpeg",
  "message": "Upload successful"
}
```

##### Error Response

| Status Code | Description                                     |
| ----------- | ----------------------------------------------- |
| 400         | Missing filename, invalid file type, empty file |
| 500         | Upload failed with server error                 |

#### Notes

- Only media files (image/_, video/_, audio/\*) are allowed by default
- Empty files are rejected
- Maximum file size is determined by server configuration

#### Example

```
POST /api/files?file=example.jpg
Content-Type: image/jpeg
Content-Length: 12345

[Binary file data]
```

---

### Delete a File

Removes a file from the server.

#### Endpoint

```
DELETE /files/:filename
```

#### URL Parameters

| Parameter | Description                    |
| --------- | ------------------------------ |
| filename  | The name of the file to delete |

#### Response

##### Success Response (200 OK)

```json
{
  "success": true,
  "fileName": "example.jpg",
  "message": "File deleted"
}
```

##### Error Response (404 Not Found)

```json
{
  "success": false,
  "message": "File not found or could not be deleted"
}
```

#### Example

```
DELETE /api/files/example.jpg
```

---

## Folder Operations

### Create a Folder

Creates a new directory.

#### Endpoint

```
POST /folders
```

#### Request Body

```json
{
  "path": "folder/path"
}
```

#### Response

##### Success Response (201 Created)

```json
{
  "success": true,
  "message": "Folder created successfully",
  "path": "folder/path"
}
```

##### Error Response

| Status Code | Description             |
| ----------- | ----------------------- |
| 400         | Missing path parameter  |
| 500         | Failed to create folder |

#### Example

```
POST /api/folders
Content-Type: application/json

{
  "path": "images/2023"
}
```

---

### List Folder Contents

Lists files and subdirectories in a specific folder.

#### Endpoint

```
GET /folders/:path
```

#### URL Parameters

| Parameter | Description                      |
| --------- | -------------------------------- |
| path      | Path to the folder (URL-encoded) |

#### Query Parameters

| Parameter | Required | Default | Description                             |
| --------- | -------- | ------- | --------------------------------------- |
| recursive | No       | false   | Whether to list files in subdirectories |

#### Response

##### Success Response (200 OK)

```json
{
  "success": true,
  "path": "images",
  "recursive": false,
  "items": [
    {
      "name": "2023",
      "path": "images/2023",
      "isDirectory": true
    },
    {
      "name": "profile.jpg",
      "path": "images/profile.jpg",
      "isDirectory": false,
      "size": 54321,
      "mtime": "2023-05-10T15:30:45Z"
    }
    // Additional items...
  ]
}
```

##### Error Response (500 Internal Server Error)

```json
{
  "success": false,
  "message": "Failed to list folder contents: folder not found"
}
```

#### Example

```
GET /api/folders/images
```

---

### Delete a Folder

Deletes a directory and optionally its contents.

#### Endpoint

```
DELETE /folders/:path
```

#### URL Parameters

| Parameter | Description                      |
| --------- | -------------------------------- |
| path      | Path to the folder (URL-encoded) |

#### Query Parameters

| Parameter | Required | Default | Description                             |
| --------- | -------- | ------- | --------------------------------------- |
| recursive | No       | false   | Whether to delete non-empty directories |

#### Response

##### Success Response (200 OK)

```json
{
  "success": true,
  "message": "Folder deleted successfully",
  "path": "images/2023",
  "recursive": true
}
```

##### Error Response

| Status Code | Description                                                          |
| ----------- | -------------------------------------------------------------------- |
| 400         | Missing path parameter or directory not empty (when recursive=false) |
| 500         | Failed to delete folder                                              |

#### Example

```
DELETE /api/folders/images%2F2023?recursive=true
```

---

## File Management Operations

### Move a File

Moves a file to a different location.

#### Endpoint

```
POST /files/move
```

#### Request Body

```json
{
  "source": "original/path/file.jpg",
  "destination": "new/path/file.jpg"
}
```

#### Response

##### Success Response (200 OK)

```json
{
  "success": true,
  "message": "File moved successfully",
  "source": "original/path/file.jpg",
  "destination": "new/path/file.jpg"
}
```

##### Error Response

| Status Code | Description                              |
| ----------- | ---------------------------------------- |
| 400         | Missing source or destination parameters |
| 500         | Failed to move file                      |

#### Example

```
POST /api/files/move
Content-Type: application/json

{
  "source": "temp/upload.jpg",
  "destination": "images/profile.jpg"
}
```

---

### Bulk Operations

Performs bulk operations on multiple files.

#### Endpoint

```
POST /files/bulk
```

> Note: This endpoint is reserved for future implementation.

#### Response

##### Current Response (501 Not Implemented)

```json
{
  "success": false,
  "message": "Bulk operations not yet implemented"
}
```

---

## Cache Management

### Cache Management

Get cache statistics or clear the cache.

#### Endpoint

```
GET /files/cache
```

#### Query Parameters

| Parameter | Required | Default | Description                      |
| --------- | -------- | ------- | -------------------------------- |
| action    | No       | "stats" | Cache action: "stats" or "clear" |

#### Request Headers

| Header      | Required | Description                  |
| ----------- | -------- | ---------------------------- |
| x-admin-key | Yes      | Admin key for authentication |

#### Response

##### Cache Stats Response (200 OK)

```json
{
  "success": true,
  "stats": {
    "size": 1234567,
    "entries": 42,
    "hits": 100,
    "misses": 20,
    "hitRate": 0.83
  }
}
```

##### Cache Clear Response (200 OK)

```json
{
  "success": true,
  "action": "clear",
  "result": {
    "entriesRemoved": 42,
    "bytesFreed": 1234567
  }
}
```

##### Error Response (403 Forbidden)

```
Unauthorized access
```

#### Example

```
GET /api/files/cache?action=clear
x-admin-key: your-admin-key
```

---

## File Format Support

### Supported File Types

#### Media Files for Upload

- Images: jpg, jpeg, png, gif, webp, svg, etc.
- Videos: mp4, mov, avi, mkv, webm, etc.
- Audio: mp3, wav, ogg, m4a, etc.

#### Compressible Content Types

The following MIME types are automatically compressed when the client supports it:

- text/plain
- text/html
- text/css
- application/javascript
- application/json
- application/xml
- image/svg+xml

---

## Advanced Features

### Range Requests

The file hosting service supports HTTP Range requests for partial content delivery, which is useful for:

- Resuming interrupted downloads
- Streaming video content
- Efficient partial file retrieval

Example:

```
GET /api/files/video.mp4
Range: bytes=1024-2047
```

Response will contain status code 206 (Partial Content) and the requested byte range.

### Conditional Requests

The service supports conditional GET requests using ETags and modification dates:

- `If-None-Match`: Returns 304 Not Modified if the file's ETag matches
- `If-Modified-Since`: Returns 304 Not Modified if the file hasn't changed since the specified date

This helps reduce bandwidth for unchanged resources.

### Content Compression

Text-based files are automatically compressed when the client supports it:

- Brotli compression (br)
- Gzip compression (gzip)
- Deflate compression (deflate)

The compression is applied based on the Accept-Encoding header sent by the client.

---

## Error Handling

All API endpoints include comprehensive error handling with appropriate HTTP status codes and descriptive error messages. Error responses include:

- HTTP status code appropriate to the error type
- JSON response with error details (when applicable)
- Descriptive error message

Common error scenarios are documented under each endpoint.

---

## Limitations and Constraints

- Maximum file size for uploads: Determined by server configuration
- File types for upload: Limited to media files by default
- Rate limiting: Depends on server configuration
- Storage limits: Determined by available server storage

---

## API Versioning

This documentation covers the current API version. The API does not include explicit version numbers in the URL.

---

## Support

For issues, questions, or feature requests related to the File Hosting Service, please contact the system administrator.

---

_Last Updated: May 7, 2025_
