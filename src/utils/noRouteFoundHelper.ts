/**
 * Helper utility for generating 404 Not Found HTML pages
 *
 * This module provides functionality to create user-friendly HTML pages
 * when no route matches the requested path, including helpful suggestions
 * for available routes.
 */

import { IncomingRequest } from '../entities/http';
import { createSSRBuilder } from 'element-crafter';
// If you need TypeScript types, run: npm install --save-dev @types/element-crafter

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

  const builder = createSSRBuilder({ escapeContent: false });

  // Styles
  const styles = includeStyles
    ? builder.createElement(
        'style',
        {},
        undefined,
        `body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; line-height: 1.6; color: #333; background-color: #f9f9f9; }
        .container { background: white; padding: 3rem; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #e74c3c; margin-bottom: 1rem; font-size: 2.5rem; }
        .error-code { font-size: 4rem; font-weight: bold; color: #e74c3c; margin: 0; line-height: 1; }
        .requested-path { background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 4px; font-family: 'Monaco', 'Consolas', monospace; border-left: 4px solid #e74c3c; margin: 1rem 0; }
        .routes-list { background: #f8f9fa; padding: 1rem; border-radius: 4px; border-left: 4px solid #3498db; margin: 1rem 0; }
        .routes-list h3 { margin-top: 0; color: #3498db; }
        .route-item { font-family: 'Monaco', 'Consolas', monospace; padding: 0.25rem 0; color: #2c3e50; cursor: pointer; }
        .route-item:hover { background: #ecf0f1; padding: 0.25rem 0.5rem; border-radius: 3px; }
        .footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee; color: #666; font-size: 0.9rem; }
        .back-link { display: inline-block; margin-top: 1rem; padding: 0.75rem 1.5rem; background: #3498db; color: white; text-decoration: none; border-radius: 4px; transition: background 0.3s; }
        .back-link:hover { background: #2980b9; }
        @media (max-width: 600px) { body { padding: 1rem; } .container { padding: 1.5rem; } h1 { font-size: 2rem; } .error-code { font-size: 3rem; } }`,
      )
    : '';

  // Custom message section
  const customMessageSection = customMessage
    ? builder.createElement(
        'div',
        {
          style:
            'background: #fff3cd; padding: 1rem; border-radius: 4px; border-left: 4px solid #ffc107; margin: 1rem 0;',
        },
        undefined,
        builder.createElement('p', { style: 'margin: 0;' }, { escapeContent: true }, customMessage),
      )
    : '';

  // Routes section
  let routesSection;
  if (availableRoutes.length > 0) {
    routesSection = builder.createElement(
      'div',
      { class: 'routes-list' },
      undefined,
      builder.createElement('h3', {}, undefined, '📍 Available Routes'),
      builder.createElement('p', {}, undefined, 'Here are some routes you might be looking for:'),
      ...availableRoutes.map((route) =>
        builder.createElement(
          'div',
          {
            class: 'route-item',
            'data-route': route,
            onclick: 'copyRouteToClipboard(event)', // Will be handled by client-side script
            title: 'Click to copy route link',
            style: 'display: flex; align-items: center; gap: 0.5em;',
          },
          undefined,
          builder.createElement('span', { style: 'flex: 1;' }, undefined, route),
          builder.createElement(
            'span',
            {
              class: 'clipboard-icon',
              style: 'font-size: 1.2em; cursor: pointer; margin-left: 0.5em;',
            },
            undefined,
            '📋',
          ),
        ),
      ),
    );
  } else {
    routesSection = builder.createElement(
      'div',
      { class: 'routes-list' },
      undefined,
      builder.createElement('h3', {}, undefined, '🔍 No Routes Available'),
      builder.createElement(
        'p',
        {},
        undefined,
        'No routes are currently registered on this server.',
      ),
    );
  }

  // Script for clipboard copy and notification
  const script = builder.createElement(
    'script',
    {},
    undefined,
    `function copyRouteToClipboard(e) {
      var route = e.currentTarget.getAttribute('data-route');
      var baseUrl = window.location.origin;
      var fullUrl = baseUrl + route;
      if (navigator.clipboard) {
        navigator.clipboard.writeText(fullUrl).then(function() {
          showCopyNotification('Copied to clipboard: ' + fullUrl);
        });
      } else {
        // fallback
        var tempInput = document.createElement('input');
        tempInput.value = fullUrl;
        document.body.appendChild(tempInput);
        tempInput.select();
        document.execCommand('copy');
        document.body.removeChild(tempInput);
        showCopyNotification('Copied to clipboard: ' + fullUrl);
      }
    }

    function showCopyNotification(message) {
      var notif = document.createElement('div');
      notif.textContent = message;
      notif.style.position = 'fixed';
      notif.style.bottom = '2rem';
      notif.style.left = '50%';
      notif.style.transform = 'translateX(-50%)';
      notif.style.background = '#3498db';
      notif.style.color = 'white';
      notif.style.padding = '0.75rem 2rem';
      notif.style.borderRadius = '6px';
      notif.style.fontSize = '1rem';
      notif.style.boxShadow = '0 2px 8px rgba(0,0,0,0.15)';
      notif.style.zIndex = '9999';
      document.body.appendChild(notif);
      setTimeout(function() {
        notif.style.opacity = '0';
        setTimeout(function() { notif.remove(); }, 400);
      }, 1500);
    }
    `,
  );

  // Build HTML
  return (
    '<!DOCTYPE html>' +
    builder.createElement(
      'html',
      { lang: 'en' },
      undefined,
      builder.createElement(
        'head',
        {},
        undefined,
        builder.createElement('meta', { charset: 'UTF-8' }),
        builder.createElement('meta', {
          name: 'viewport',
          content: 'width=device-width, initial-scale=1.0',
        }),
        builder.createElement('title', {}, undefined, title),
        styles,
      ),
      builder.createElement(
        'body',
        {},
        undefined,
        builder.createElement(
          'div',
          { class: 'container' },
          undefined,
          builder.createElement('div', { class: 'error-code' }, undefined, '404'),
          builder.createElement('h1', {}, undefined, heading),
          builder.createElement(
            'p',
            {},
            undefined,
            'The requested page could not be found on this server.',
          ),
          builder.createElement(
            'div',
            { class: 'requested-path' },
            undefined,
            builder.createElement('strong', {}, undefined, 'Requested:'),
            ' ' + requestedPath,
          ),
          customMessageSection,
          routesSection,
          builder.createElement('a', { href: '/', class: 'back-link' }, undefined, '← Go to Home'),
          builder.createElement(
            'div',
            { class: 'footer' },
            undefined,
            builder.createElement(
              'p',
              {},
              undefined,
              builder.createElement('strong', {}, undefined, serverName),
              ' • Generated at ' + new Date().toLocaleString(),
            ),
          ),
          script,
        ),
      ),
    )
  );
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
