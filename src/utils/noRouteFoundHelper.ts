/**
 * Helper utility for generating 404 Not Found HTML pages
 *
 * This module provides functionality to create user-friendly HTML pages
 * when no route matches the requested path, including helpful suggestions
 * for available routes.
 */

import { IncomingRequest } from '../entities/http';

/**
 * Options for customizing the 404 HTML page
 */
export interface NoRouteFoundOptions {
  /** The requested path that was not found */
  requestedPath: string;
  /** Array of available route paths to suggest to the user */
  availableRoutes?: string[];
  /** Custom title for the page (default: "404 - Page Not Found") */
  title?: string;
  /** Custom heading for the page (default: "Page Not Found") */
  heading?: string;
  /** Additional custom message to display */
  customMessage?: string;
  /** Whether to include CSS styling (default: true) */
  includeStyles?: boolean;
  /** Server name or application name to display */
  serverName?: string;
}

/**
 * Generates a complete HTML page for 404 Not Found responses
 *
 * @param options Configuration options for the HTML page
 * @returns Complete HTML string ready to be sent as response body
 */
function generateNoRouteFoundHtml(options: NoRouteFoundOptions): string {
  const {
    requestedPath,
    availableRoutes = [],
    title = '404 - Page Not Found',
    heading = 'Page Not Found',
    customMessage,
    includeStyles = true,
    serverName = 'My Central Hub',
  } = options;

  const styles = includeStyles
    ? `
    <style>
      body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
        line-height: 1.6;
        color: #333;
        background-color: #f9f9f9;
      }
      .container {
        background: white;
        padding: 3rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      }
      h1 {
        color: #e74c3c;
        margin-bottom: 1rem;
        font-size: 2.5rem;
      }
      .error-code {
        font-size: 4rem;
        font-weight: bold;
        color: #e74c3c;
        margin: 0;
        line-height: 1;
      }
      .requested-path {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-family: 'Monaco', 'Consolas', monospace;
        border-left: 4px solid #e74c3c;
        margin: 1rem 0;
      }
      .routes-list {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
      }
      .routes-list h3 {
        margin-top: 0;
        color: #3498db;
      }
      .route-item {
        font-family: 'Monaco', 'Consolas', monospace;
        padding: 0.25rem 0;
        color: #2c3e50;
      }
      .route-item:hover {
        background: #ecf0f1;
        padding: 0.25rem 0.5rem;
        border-radius: 3px;
        cursor: pointer;
      }
      .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
        color: #666;
        font-size: 0.9rem;
      }
      .back-link {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.75rem 1.5rem;
        background: #3498db;
        color: white;
        text-decoration: none;
        border-radius: 4px;
        transition: background 0.3s;
      }
      .back-link:hover {
        background: #2980b9;
      }
      @media (max-width: 600px) {
        body { padding: 1rem; }
        .container { padding: 1.5rem; }
        h1 { font-size: 2rem; }
        .error-code { font-size: 3rem; }
      }
    </style>
  `
    : '';

  const routesSection =
    availableRoutes.length > 0
      ? `
    <div class="routes-list">
      <h3>📍 Available Routes</h3>
      <p>Here are some routes you might be looking for:</p>
      ${availableRoutes.map((route) => `<div class="route-item">${route}</div>`).join('')}
    </div>
  `
      : `
    <div class="routes-list">
      <h3>🔍 No Routes Available</h3>
      <p>No routes are currently registered on this server.</p>
    </div>
  `;

  const customMessageSection = customMessage
    ? `
    <div style="background: #fff3cd; padding: 1rem; border-radius: 4px; border-left: 4px solid #ffc107; margin: 1rem 0;">
      <p style="margin: 0;">${customMessage}</p>
    </div>
  `
    : '';

  return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title}</title>
    ${styles}
</head>
<body>
    <div class="container">
        <div class="error-code">404</div>
        <h1>${heading}</h1>
        
        <p>The requested page could not be found on this server.</p>
        
        <div class="requested-path">
            <strong>Requested:</strong> ${requestedPath}
        </div>

        ${customMessageSection}
        
        ${routesSection}
        
        <a href="/" class="back-link">← Go to Home</a>
        
        <div class="footer">
            <p>
                <strong>${serverName}</strong> • 
                Generated at ${new Date().toLocaleString()}
            </p>
        </div>
    </div>
</body>
</html>`;
}

/**
 * Convenience function to generate 404 HTML from a request object
 *
 * @param req The incoming request object
 * @param availableRoutes Array of available route paths
 * @param customOptions Additional options to override defaults
 * @returns Complete HTML string ready to be sent as response body
 */
export function generateNoRouteFoundHtmlFromRequest(
  req: IncomingRequest,
  availableRoutes: string[] = [],
  customOptions: Partial<NoRouteFoundOptions> = {},
): string {
  return generateNoRouteFoundHtml({
    requestedPath: req.path,
    availableRoutes,
    ...customOptions,
  });
}
