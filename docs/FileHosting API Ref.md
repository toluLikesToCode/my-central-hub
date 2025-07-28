# ✨ File Hosting API Reference

Welcome to the My Central Hub File Hosting API! This guide focuses on how to list your files and retrieve detailed statistics about them. Let's explore!

---

## List Files

Need to see what files are available? This endpoint returns a paginated list of files stored on the server, allowing you to filter and sort the results to find exactly what you need.

> Fun Fact: Pagination prevents overwhelming you (and the server!) with potentially thousands of files at once. Smart, right?

### Request

`GET /api/files`

#### Query Parameters

| Parameter  | Type    | Required | Default | Description                                                                                                                                                                                              |
| :--------- | :------ | :------- | :------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `page`     | integer | No       | `1`     | The page number you want to retrieve (for pagination).                                                                                                                                                   |
| `limit`    | integer | No       | `20`    | How many files to show per page (max value might be limited).                                                                                                                                            |
| `sort`     | string  | No       | `name`  | Field to sort by (`fileName`, `size`, `lastModified`, `mimeType`).                                                                                                                                       |
| `order`    | string  | No       | `asc`   | Sorting direction (`asc` for ascending, `desc` descending).                                                                                                                                              |
| `type`     | string  | No       |         | Filter by MIME type prefix (e.g., `image/`, `video/mp4`).                                                                                                                                                |
| `search`   | string  | No       |         | Simple substring search on filenames and MIME types.                                                                                                                                                     |
| `filter`   | string  | No       |         | JSON-encoded advanced filter expression supporting `and`, `or`, `not`, `regex`, and basic operators (`fileName`, `mimeType`, `minSize`, `maxSize`, `dateFrom`, `dateTo`, `extension`, dimensions, etc.). |
| `dateFrom` | string  | No       |         | Filter files modified _after_ this ISO date (e.g., `2023-01-01`).                                                                                                                                        |
| `dateTo`   | string  | No       |         | Filter files modified _before_ this ISO date (e.g., `2023-05-10`).                                                                                                                                       |
| `sizeFrom` | integer | No       | `0`     | Minimum file size in bytes.                                                                                                                                                                              |
| `sizeTo`   | integer | No       |         | Maximum file size in bytes (no upper limit when omitted).                                                                                                                                                |

### Response

Returns a JSON object containing the list of files and pagination details.

#### Response Schema (`200 OK`)

```json
{
  "files": [
    {
      "name": "string",
      "path": "string",
      "size": integer,
      "mimeType": "string",
      "lastModified": "string",
      "url": "string"
    }
    // ... more file objects
  ],
  "pagination": {
    "page": integer,
    "limit": integer,
    "totalFiles": integer,
    "totalPages": integer,
    "hasNextPage": boolean,
    "hasPrevPage": boolean
  },
  "_links": {
    "self": "string",
    "first": "string",
    "last": "string",
    "next": "string | null",
    "prev": "string | null"
  }
}
```

#### Error Response Schema (`4xx` or `5xx`)

```json
{
  "success": false,
  "message": "string" // Description of the error
  // Additional error details may be included for 5xx errors
}
```

### Quick Example (Curl)

Get the first 10 image files, sorted by size descending:

```bash
curl "http://localhost:8080/api/files?limit=10&type=image/&sort=size&order=desc"
```

### Advanced Filtering Example (Curl)

Return only video files excluding any filename containing "thumbnail":

```bash
curl "http://localhost:8080/api/files?filter={\"and\":[{\"mimeType\":\"video/\"},{\"not\":{\"fileName\":\"thumbnail\"}}]}"
```

_Note: When using this in a shell, URL-encode the JSON or wrap it in single quotes to avoid quoting issues._

### Real-World Example (JavaScript Fetch)

Let's build a simple image gallery that loads images page by page.

```javascript
async function loadImageGalleryPage(page = 1) {
  const galleryElement = document.getElementById('image-gallery');
  const loadMoreButton = document.getElementById('load-more');
  const statusElement = document.getElementById('status');

  statusElement.textContent = 'Loading images...';
  loadMoreButton.disabled = true;

  try {
    const response = await fetch(
      `/api/files?page=${page}&limit=12&type=image/&sort=date&order=desc`
    );
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();

    data.files.forEach(file => {
      const imgElement = document.createElement('img');
      imgElement.src = file.url; // Use the direct file URL
      imgElement.alt = file.name;
      imgElement.title = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      galleryElement.appendChild(imgElement);
    });

    statusElement.textContent = `Showing page ${data.pagination.page} of ${data.pagination.totalPages}. Total images: ${data.pagination.totalFiles}`;

    // Enable or disable the "Load More" button
    if (data.pagination.hasNextPage) {
      loadMoreButton.disabled = false;
      loadMoreButton.dataset.nextPage = data.pagination.page + 1; // Store next page number
    } else {
      loadMoreButton.style.display = 'none'; // Hide if no more pages
      statusElement.textContent += ' (End of list)';
    }

  } catch (error) {
    console.error('Failed to load images:', error);
    statusElement.textContent = `Error loading images: ${error.message}`;
    loadMoreButton.disabled = false; // Re-enable button on error
  }
}

// Initial load
loadImageGalleryPage(1);

// Setup button listener
document.getElementById('load-more')?.addEventListener('click', (event) => {
  const nextPage = parseInt((event.target as HTMLButtonElement).dataset.nextPage || '1');
  loadImageGalleryPage(nextPage);
});
```

---

## Retrieve File Statistics

Curious about the nitty-gritty details of a specific file? This endpoint provides in-depth metadata, including dimensions, duration, codecs, and more.

> Perfect for when you need to know if that video is 1080p or 4K before deciding to stream it\!

### Request

`GET /api/files/:filename/stats`

#### URL Parameters

| Parameter | Type   | Required | Description                                          |
| :-------- | :----- | :------- | :--------------------------------------------------- |
| filename  | string | Yes      | The URL-encoded name (or path) of the file to query. |

### Response

Returns a JSON object containing detailed statistics for the requested file.

#### Response Schema (`200 OK`)

```json
{
  "success": true,
  "fileName": "string", // The requested filename
  "stats": {
    "id": "integer | null", // Database ID (if available)
    "fileName": "string",
    "filePath": "string", // Relative path from the root
    "mimeType": "string", // e.g., "video/mp4"
    "size": "integer", // Size in bytes
    "lastModified": "string", // ISO 8601 date string
    "width": "integer | null", // Image/video width in pixels
    "height": "integer | null", // Image/video height in pixels
    "duration": "number | null", // Video/audio duration in seconds
    "bitrate": "integer | null", // Average bitrate in bits per second
    "encoding": "string | null", // e.g., "h264"
    "codec": "string | null", // Specific codec name
    "frameRate": "number | null", // Frames per second (for video)
    "audioChannels": "integer | null", // Number of audio channels
    "sampleRate": "integer | null", // Audio sample rate in Hz
    "createdAt": "string", // ISO 8601 date string (when stats were first recorded)
    "updatedAt": "string", // ISO 8601 date string (when stats were last updated)
    // --- Human-Readable Formats ---
    "sizeFormatted": "string", // e.g., "12.34 MB"
    "lastModifiedFormatted": "string", // e.g., "May 08, 2025 07:33:16 PM PDT"
    "durationFormatted": "string | undefined" // e.g., "1:30.500" or "0:15.123"
  },
  "_links": {
    "self": "string", // URL for this stats request
    "file": "string", // URL to retrieve the actual file content
    "head": {
      "href": "string", // URL to get file metadata via HEAD request
      "method": "HEAD",
      "description": "string" // Explains what the HEAD request does
    }
  }
}
```

#### Error Response Schema (`400`, `404`, `500`)

```json
{
  "success": false,
  "message": "string" // Description of the error (e.g., "File not found", "Missing filename")
}
```

### Quick Example (Curl)

Get stats for a video file named `epic_moment.mp4`:

```bash
curl "http://localhost:8080/api/files/epic_moment.mp4/stats"
```

### Real-World Example (JavaScript Fetch)

Display detailed information about a media file when a user clicks on it in a list.

```javascript
async function showFileDetails(fileName) {
  const detailsPanel = document.getElementById('details-panel');
  const statusElement = document.getElementById('details-status');

  statusElement.textContent = 'Loading details...';
  detailsPanel.innerHTML = ''; // Clear previous details

  try {
    // Encode the filename in case it contains special characters
    const encodedFileName = encodeURIComponent(fileName);
    const response = await fetch(`/api/files/${encodedFileName}/stats`);

    if (response.status === 404) {
       throw new Error(`File not found: ${fileName}`);
    }
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`API error (${response.status}): ${errorData.message || 'Failed to fetch stats'}`);
    }

    const data = await response.json();
    const stats = data.stats;

    // Build HTML to display the stats
    let html = `<h3>${stats.fileName}</h3>`;
    html += `<ul>`;
    html += `<li><strong>Path:</strong> ${stats.filePath}</li>`;
    html += `<li><strong>Type:</strong> ${stats.mimeType}</li>`;
    html += `<li><strong>Size:</strong> ${stats.sizeFormatted} (${stats.size} bytes)</li>`;
    html += `<li><strong>Modified:</strong> ${stats.lastModifiedFormatted}</li>`;
    if (stats.width && stats.height) {
      html += `<li><strong>Dimensions:</strong> ${stats.width} x ${stats.height}</li>`;
    }
    if (stats.durationFormatted) {
      html += `<li><strong>Duration:</strong> ${stats.durationFormatted}</li>`;
    }
    if (stats.codec) {
      html += `<li><strong>Codec:</strong> ${stats.codec}</li>`;
    }
    if (stats.frameRate) {
      html += `<li><strong>Frame Rate:</strong> ${stats.frameRate.toFixed(2)} fps</li>`;
    }
    if (stats.bitrate) {
      html += `<li><strong>Bitrate:</strong> ${(stats.bitrate / 1000).toFixed(0)} kbps</li>`;
    }
    if (stats.audioChannels) {
      html += `<li><strong>Audio Channels:</strong> ${stats.audioChannels}</li>`;
    }
    if (stats.sampleRate) {
      html += `<li><strong>Sample Rate:</strong> ${stats.sampleRate / 1000} kHz</li>`;
    }
    html += `</ul>`;
    html += `<p><a href="${data._links.file}" target="_blank">View/Download File</a></p>`;

    detailsPanel.innerHTML = html;
    statusElement.textContent = 'Details loaded.';

  } catch (error) {
    console.error('Failed to load file details:', error);
    statusElement.textContent = `Error: ${error.message}`;
    detailsPanel.innerHTML = `<p style="color: red;">Could not load details for ${fileName}.</p>`;
  }
}

// Example usage: Assume you have a list item with data-filename attribute
document.querySelectorAll('.file-list-item').forEach(item => {
  item.addEventListener('click', () => {
    const fileName = (item as HTMLElement).dataset.filename;
    if (fileName) {
      showFileDetails(fileName);
    }
  });
});
```

---

## ✨ Best Practices

Follow these tips for optimal performance and security when interacting with the File Hosting API.

### Performance

1.  **Use Pagination:** Always use the `page` and `limit` parameters when listing files (`/api/files`) to avoid requesting huge amounts of data at once. Start with a reasonable `limit` (e.g., 20-50) and increase if needed.
2.  **Leverage Filtering & Sorting:** Utilize server-side filtering (`type`, `search`, `dateFrom`/`To`, `sizeFrom`/`To`) and sorting (`sort`, `order`) for the `/api/files` endpoint. This is much more efficient than fetching all files and filtering/sorting on the client side.
3.  **Utilize `HEAD` Requests:** Before downloading a large file, consider sending a `HEAD` request to `/api/files/:filename` first. This retrieves only the headers (including `Content-Length`, `Last-Modified`, `ETag`) without the file body, allowing you to check the size or modification status efficiently.
4.  **Conditional Requests (`GET`):** Use `If-None-Match` (with the `ETag` from a previous request) or `If-Modified-Since` (with the `Last-Modified` date) headers when retrieving files. If the file hasn't changed, the server will respond with a `304 Not Modified` status and an empty body, saving bandwidth.
5.  **Range Requests (`GET`):** For large files, especially video or audio, use the `Range: bytes=start-end` header to request only the necessary chunks. This enables features like resuming downloads and seeking in media players.
6.  **Client-Side Caching:** Respect the `Cache-Control`, `ETag`, and `Last-Modified` headers provided by the server. Browser caches or client-side storage can significantly reduce redundant requests for static assets.

### Security

1.  **Validate Filenames/Paths:** While the server attempts to sanitize paths, clients should still perform basic validation on filenames or paths received from user input before sending them to the API to prevent unexpected behavior. Avoid constructing paths with `../`.
2.  **Handle Errors Gracefully:** Always check the HTTP status code and the `success` field in the JSON response. Implement robust error handling in your client to inform the user if an operation fails (e.g., file not found, server error).
3.  **Rate Limiting (Server-Side):** Be aware that the server might implement rate limiting. If you receive `429 Too Many Requests` errors, reduce the frequency of your API calls.
4.  **Admin Operations:** Operations like cache clearing (`/api/files/cache?action=clear`) require an admin key. Securely manage this key and ensure it's only used by authorized administrative tools, never exposed in client-side code.

<!-- end list -->

```

```
