# API Reference: GET /api/files (List Files)

This endpoint allows you to retrieve a paginated, sortable, and filterable list of files managed by the My Central Hub file hosting service.

## Endpoint

`GET /api/files`

## Description

Returns a list of files based on the specified query parameters. The response includes pagination details to navigate through large result sets and HATEOAS links for discovering related actions.

## Query Parameters

All query parameters are optional.

| Parameter  | Type    | Default | Description                                                                                                                                             | Example                                                                    |
| :--------- | :------ | :------ | :------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------- |
| `page`     | integer | `1`     | The page number of the results to retrieve.                                                                                                             | `?page=2`                                                                  |
| `limit`    | integer | `20`    | The maximum number of files to return per page.                                                                                                         | `?limit=50`                                                                |
| `sort`     | string  | `name`  | The field to sort the files by. Valid values: `name` (aliases to `fileName`), `fileName`, `size`, `lastModified`, `mimeType`, `createdAt`, `updatedAt`. | `?sort=size`                                                               |
| `order`    | string  | `asc`   | The order to sort the files in. Valid values: `asc` (ascending), `desc` (descending).                                                                   | `?order=desc`                                                              |
| `type`     | string  |         | Filters files by MIME type. Can be a full MIME type (e.g., `image/jpeg`) or a prefix (e.g., `image/` to get all images, `video/` for all videos).       | `?type=image/png` or `?type=video/`                                        |
| `search`   | string  |         | Performs a case-insensitive substring search on `fileName` and `mimeType` fields.                                                                       | `?search=report`                                                           |
| `dateFrom` | string  |         | Filters files modified on or after this date. Expected format: ISO 8601 Date string (e.g., `YYYY-MM-DD`).                                               | `?dateFrom=2023-01-01`                                                     |
| `dateTo`   | string  |         | Filters files modified on or before this date. Expected format: ISO 8601 Date string (e.g., `YYYY-MM-DD`).                                              | `?dateTo=2023-12-31`                                                       |
| `sizeFrom` | integer | `0`     | Filters files with a size greater than or equal to this value (in bytes).                                                                               | `?sizeFrom=1048576` (1MB)                                                  |
| `sizeTo`   | integer |         | Filters files with a size less than or equal to this value (in bytes). If omitted or `0`, there is no upper size limit.                                 | `?sizeTo=5242880` (5MB)                                                    |
| `filter`   | string  |         | A URL-encoded JSON string representing a complex filter object. See [Advanced Filtering](#advanced-filtering) for details.                              | `?filter={"and":[{"mimeType":"image/"},{"not":{"fileName":"thumbnail"}}]}` |

---

## Advanced Filtering (`filter` parameter)

The `filter` query parameter accepts a URL-encoded JSON string that defines a structured filter. This allows for more complex queries than the simple top-level parameters.

### Filter Object Structure:

```typescript
interface FileFilter {
  // Boolean logic
  not?: FileFilter; // Negates the inner filter
  and?: FileFilter[]; // All inner filters must match
  or?: FileFilter[]; // At least one inner filter must match

  // Pattern matching
  regex?: { [key in keyof FileStats]?: string }; // Property name to regex pattern string

  // Simple equality / range clauses on FileStats properties
  fileName?: string; // Substring match on file name
  mimeType?: string; // Substring match on MIME type
  extension?: string; // Exact match on file extension (e.g., "mp4", "jpg")
  minSize?: number; // Minimum file size in bytes
  maxSize?: number; // Maximum file size in bytes
  minWidth?: number; // Minimum image/video width in pixels
  minHeight?: number; // Minimum image/video height in pixels
  minDuration?: number; // Minimum video/audio duration in seconds
  dateFrom?: string | Date; // Last modified on or after this date (ISO string)
  dateTo?: string | Date; // Last modified on or before this date (ISO string)
}
```

Note: FileStats refers to the internal representation of file metadata. The available fields for regex and direct property filters include fileName, filePath, mimeType, size, lastModified (as a Date object for comparison, or its ISO string representation), width, height, duration, codec, encoding, etc.

### Example Complex Filter:

To find all video files that are larger than 10MB OR images that are not JPEGs and were modified in 2023:

```json
{
  "or": [
    {
      "and": [
        { "mimeType": "video/" },
        { "minSize": 10485760 } // 10MB
      ]
    },
    {
      "and": [
        { "mimeType": "image/" },
        { "not": { "mimeType": "image/jpeg" } },
        { "dateFrom": "2023-01-01T00:00:00.000Z" },
        { "dateTo": "2023-12-31T23:59:59.999Z" }
      ]
    }
  ]
}
```

When URL-encoded for the filter parameter:

```json
%7B%22or%22%3A%5B%7B%22and%22%3A%5B%7B%22mimeType%22%3A%22video%2F%22%7D%2C%7B%22minSize%22%3A10485760%7D%5D%7D%2C%7B%22and%22%3A%5B%7B%22mimeType%22%3A%22image%2F%22%7D%2C%7B%22not%22%3A%7B%22mimeType%22%3A%22image%2Fjpeg%22%7D%7D%2C%7B%22dateFrom%22%3A%222023-01-01T00%3A00%3A00.000Z%22%7D%2C%7B%22dateTo%22%3A%222023-12-31T23%3A59%3A59.999Z%22%7D%5D%7D%5D%7D
```

## Responses

### Success Response (200 OK)

**Content-Type:** `application/json`

**Body:**

```json
{
  "files": [
    {
      "name": "example.jpg",
      "path": "images/example.jpg",
      "size": 102400,
      "mimeType": "image/jpeg",
      "lastModified": "May 14, 2025 03:45:00 PM PDT",
      "width": 1920,
      "height": 1080,
      "url": "/api/files/images%2Fexample.jpg",
      "id": 123
    }
    // ... more file objects
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "totalFiles": 150,
    "totalPages": 8,
    "hasNextPage": true,
    "hasPrevPage": false
  },
  "_links": {
    "self": "/api/files?page=1&limit=20",
    "next": "/api/files?page=2&limit=20",
    "prev": null
  }
}
```

### File Object Properties:

- **name (string):** The name of the file (e.g., example.jpg).
- **path (string):** The relative path to the file from the media root (e.g., images/example.jpg).
- **size (integer):** The size of the file in bytes.
- **mimeType (string):** The MIME type of the file (e.g., image/jpeg).
- **lastModified (string):** A human-readable string of the last modification date and time (e.g., May 14, 2025 03:45:00 PM PDT). The exact format depends on server configuration.
- **width (integer | string):** The width of the image/video in pixels, or "N/A" if not applicable/available.
- **height (integer | string):** The height of the image/video in pixels, or "N/A" if not applicable/available.
- **url (string):** A relative URL to download or access the specific file (e.g., /api/files/images%2Fexample.jpg). Note that the path component is URL-encoded.
- **id (integer):** The internal database ID of the file's statistics record.

### Pagination Object Properties:

- **page (integer):** The current page number.
- **limit (integer):** The number of items per page.
- **totalFiles (integer):** The total number of files matching the query.
- **totalPages (integer):** The total number of pages available.
- **hasNextPage (boolean):** true if there is a next page, false otherwise.
- **hasPrevPage (boolean):** true if there is a previous page, false otherwise.

### Links Object Properties (`_links`):

- **self (string):** The URL of the current request.
- **next (string | null):** The URL for the next page of results, or null if this is the last page.
- **prev (string | null):** The URL for the previous page of results, or null if this is the first page.

## Error Responses

### 400 Bad Request:

If query parameters are invalid (e.g., non-integer page or limit, invalid filter JSON).

**Content-Type:** `application/json`

**Body:**

```json
{
  "error": "Bad Request",
  "message": "Invalid query parameter: <parameter_name>"
}
```

### 500 Internal Server Error:

If an unexpected error occurs on the server while processing the request (e.g., database error).

**Content-Type:** `application/json`

**Body:**

```json
{
  "error": "Internal server error"
}
```

## Examples

1. Get the first page of files (default limit 20, sorted by name ascending):

```bash
curl "http://localhost:8080/api/files"
```

2. Get the second page of video files, 10 per page, sorted by size descending:

```bash
curl "http://localhost:8080/api/files?page=2&limit=10&type=video/&sort=size&order=desc"
```

3. Search for files containing "vacation" in their name or MIME type:

```bash
curl "http://localhost:8080/api/files?search=vacation"
```

4. Get PNG images modified in the year 2024:

```bash
curl "http://localhost:8080/api/files?type=image/png&dateFrom=2024-01-01&dateTo=2024-12-31"
```

5. Get files larger than 5MB using the advanced filter parameter:

URL-encoded JSON: `{"minSize":5242880}` becomes `%7B%22minSize%22%3A5242880%7D`

```bash
curl "http://localhost:8080/api/files?filter=%7B%22minSize%22%3A5242880%7D"
```

6. Get MP4 videos that are not in the 'thumbnails' subfolder:

URL-encoded JSON: `{"and":[{"mimeType":"video/mp4"},{"not":{"fileName":"thumbnail"}}]}` becomes `%7B%22and%22%3A%5B%7B%22mimeType%22%3A%22video%2Fmp4%22%7D%2C%7B%22not%22%3A%7B%22fileName%22%3A%22thumbnail%22%7D%7D%5D%7D`

```bash
curl "http://localhost:8080/api/files?filter=%7B%22and%22%3A%5B%7B%22mimeType%22%3A%22video%2Fmp4%22%7D%2C%7B%22not%22%3A%7B%22fileName%22%3A%22thumbnail%22%7D%7D%5D%7D"
```

## JavaScript Fetch Example

```javascript
async function fetchFiles(page = 1, limit = 10) {
  const params = new URLSearchParams({
    page: page.toString(),
    limit: limit.toString(),
    sort: 'lastModified',
    order: 'desc',
    type: 'image/', // Fetch only images
  });

  try {
    const response = await fetch(`/api/files?${params.toString()}`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(
        `API Error (${response.status}): ${errorData.message || 'Failed to fetch files'}`,
      );
    }
    const data = await response.json();
    console.log('Files:', data.files);
    console.log('Pagination:', data.pagination);
    console.log('Links:', data._links);
    return data;
  } catch (error) {
    console.error('Error fetching files:', error);
  }
}

// Example usage:
fetchFiles(1, 5).then((data) => {
  if (data && data.files.length > 0) {
    // Display files
  }
});
```

## Additional JavaScript Examples

### Example 1: Load and Display Files with Error Handling

```javascript
async function loadFiles(options = {}) {
  const { page = 1, limit = 20, fileType = '', sortBy = 'lastModified', order = 'desc' } = options;

  const statusElement = document.getElementById('status');
  const fileListElement = document.getElementById('file-list');

  statusElement.textContent = 'Loading files...';

  try {
    const params = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
      sort: sortBy,
      order: order,
    });

    if (fileType) {
      params.append('type', fileType);
    }

    const response = await fetch(`/api/files?${params.toString()}`);

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();

    // Clear previous content
    fileListElement.innerHTML = '';

    if (data.files.length === 0) {
      fileListElement.innerHTML = '<p>No files found matching your criteria.</p>';
      statusElement.textContent = 'No files found.';
      return;
    }

    // Create file list
    const ul = document.createElement('ul');
    ul.className = 'file-list';

    data.files.forEach((file) => {
      const li = document.createElement('li');
      li.className = 'file-item';

      const fileSize = formatFileSize(file.size);
      const fileIcon = getFileIcon(file.mimeType);

      li.innerHTML = `
        <div class="file-icon">${fileIcon}</div>
        <div class="file-info">
          <div class="file-name">${file.name}</div>
          <div class="file-meta">
            <span>${file.mimeType}</span>
            <span>${fileSize}</span>
            <span>${file.lastModified}</span>
          </div>
        </div>
      `;

      li.addEventListener('click', () => {
        window.location.href = file.url;
      });

      ul.appendChild(li);
    });

    fileListElement.appendChild(ul);

    // Update pagination info
    statusElement.textContent = `Showing ${data.files.length} of ${data.pagination.totalFiles} files (Page ${data.pagination.page} of ${data.pagination.totalPages})`;

    // Set up pagination controls if needed
    setupPagination(data.pagination, options);
  } catch (error) {
    console.error('Error loading files:', error);
    statusElement.textContent = `Error: ${error.message}`;
    fileListElement.innerHTML =
      '<p class="error">Failed to load files. Please try again later.</p>';
  }
}

// Helper functions
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getFileIcon(mimeType) {
  // Simple icon mapping based on mime type
  if (mimeType.startsWith('image/')) return '🖼️';
  if (mimeType.startsWith('video/')) return '🎬';
  if (mimeType.startsWith('audio/')) return '🎵';
  if (mimeType.includes('pdf')) return '📄';
  return '📁';
}

function setupPagination(pagination, currentOptions) {
  const paginationElement = document.getElementById('pagination');
  paginationElement.innerHTML = '';

  if (pagination.totalPages <= 1) return;

  const paginationNav = document.createElement('nav');
  paginationNav.className = 'pagination-nav';

  // Previous button
  const prevButton = document.createElement('button');
  prevButton.textContent = '← Previous';
  prevButton.disabled = !pagination.hasPrevPage;
  prevButton.addEventListener('click', () => {
    loadFiles({ ...currentOptions, page: pagination.page - 1 });
  });

  // Next button
  const nextButton = document.createElement('button');
  nextButton.textContent = 'Next →';
  nextButton.disabled = !pagination.hasNextPage;
  nextButton.addEventListener('click', () => {
    loadFiles({ ...currentOptions, page: pagination.page + 1 });
  });

  paginationNav.appendChild(prevButton);
  paginationNav.appendChild(nextButton);
  paginationElement.appendChild(paginationNav);
}

// Example usage
document.addEventListener('DOMContentLoaded', () => {
  // Initialize with default options
  loadFiles();

  // Set up filter form
  document.getElementById('filter-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const fileType = document.getElementById('file-type-select').value;
    loadFiles({ fileType, page: 1 }); // Reset to page 1 when filtering
  });
});
```

### Example 2: Advanced Filter Builder

```javascript
class FileFilterBuilder {
  constructor() {
    this.filter = {};
  }

  // Simple conditions
  withFileName(name) {
    this.filter.fileName = name;
    return this;
  }

  withMimeType(type) {
    this.filter.mimeType = type;
    return this;
  }

  withExtension(ext) {
    this.filter.extension = ext.startsWith('.') ? ext.substring(1) : ext;
    return this;
  }

  withMinSize(bytes) {
    this.filter.minSize = bytes;
    return this;
  }

  withMaxSize(bytes) {
    this.filter.maxSize = bytes;
    return this;
  }

  modifiedAfter(date) {
    this.filter.dateFrom = date instanceof Date ? date.toISOString() : date;
    return this;
  }

  modifiedBefore(date) {
    this.filter.dateTo = date instanceof Date ? date.toISOString() : date;
    return this;
  }

  // Regex matching
  withFileNameMatching(pattern) {
    this.filter.regex = this.filter.regex || {};
    this.filter.regex.fileName = pattern;
    return this;
  }

  // Size helpers
  withMinSizeKB(kb) {
    return this.withMinSize(kb * 1024);
  }

  withMinSizeMB(mb) {
    return this.withMinSize(mb * 1024 * 1024);
  }

  withMaxSizeKB(kb) {
    return this.withMaxSize(kb * 1024);
  }

  withMaxSizeMB(mb) {
    return this.withMaxSize(mb * 1024 * 1024);
  }

  // Combinators
  not() {
    const currentFilter = { ...this.filter };
    this.filter = { not: currentFilter };
    return this;
  }

  and(otherBuilder) {
    const currentFilter = { ...this.filter };
    const otherFilter =
      otherBuilder instanceof FileFilterBuilder ? otherBuilder.build() : otherBuilder;

    if (currentFilter.and) {
      // If we already have an AND, add to it
      currentFilter.and.push(otherFilter);
      this.filter = currentFilter;
    } else {
      // Create a new AND
      this.filter = { and: [currentFilter, otherFilter] };
    }
    return this;
  }

  or(otherBuilder) {
    const currentFilter = { ...this.filter };
    const otherFilter =
      otherBuilder instanceof FileFilterBuilder ? otherBuilder.build() : otherBuilder;

    if (currentFilter.or) {
      // If we already have an OR, add to it
      currentFilter.or.push(otherFilter);
      this.filter = currentFilter;
    } else {
      // Create a new OR
      this.filter = { or: [currentFilter, otherFilter] };
    }
    return this;
  }

  // Get the built filter
  build() {
    return { ...this.filter };
  }

  // Get URL-encoded filter string
  buildEncoded() {
    return encodeURIComponent(JSON.stringify(this.build()));
  }

  // Build URL with parameters
  buildUrl(baseUrl = '/api/files', additionalParams = {}) {
    const params = new URLSearchParams(additionalParams);
    params.append('filter', JSON.stringify(this.build()));
    return `${baseUrl}?${params.toString()}`;
  }
}

// Example usage:
async function searchWithAdvancedFilter() {
  try {
    // Build a complex filter: (large videos OR recent images) AND not containing "temp"
    const videoFilter = new FileFilterBuilder().withMimeType('video/').withMinSizeMB(100);

    const imageFilter = new FileFilterBuilder().withMimeType('image/').modifiedAfter('2023-01-01');

    const combinedFilter = new FileFilterBuilder()
      .or(videoFilter)
      .or(imageFilter)
      .and(new FileFilterBuilder().withFileNameMatching('temp').not());

    // Additional parameters
    const params = {
      page: '1',
      limit: '50',
      sort: 'lastModified',
      order: 'desc',
    };

    // Build the URL with the filter and additional parameters
    const url = combinedFilter.buildUrl('/api/files', params);

    // Make the request
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }

    const data = await response.json();
    console.log(`Found ${data.pagination.totalFiles} files matching the criteria`);
    console.log('First few results:', data.files.slice(0, 3));

    return data;
  } catch (error) {
    console.error('Error searching files:', error);
    throw error;
  }
}

// Execute the search when needed
document.getElementById('advanced-search-btn').addEventListener('click', () => {
  searchWithAdvancedFilter()
    .then((data) => {
      displayResults(data.files);
      updatePagination(data.pagination);
    })
    .catch((err) => {
      showErrorMessage(`Search failed: ${err.message}`);
    });
});
```

These examples showcase more advanced usage of the file listing API with proper error handling, UI integration, and complex filtering capabilities.
