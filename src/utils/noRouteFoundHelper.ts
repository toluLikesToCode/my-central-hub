/**
 * Helper utility for generating 404 Not Found HTML pages
 *
 * This module provides functionality to create user-friendly HTML pages
 * when no route matches the requested path, including helpful suggestions
 * for available routes.
 */

import { IncomingRequest } from '../entities/http';
import { createSSRBuilder, PageBuilder } from 'element-crafter';
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

  const builder = createSSRBuilder({ escapeContent: true });

  // HEAD content (avoid createFragment to prevent double-escaping in SSR)
  let head = '';
  head += String(builder.createElement('meta', { charset: 'UTF-8' }));
  head += String(
    builder.createElement('meta', {
      name: 'viewport',
      content: 'width=device-width, initial-scale=1.0',
    }),
  );
  if (includeStyles) {
    head += String(
      builder.createElement(
        'style',
        {},
        undefined,
        `:root{--accent:#3498db;--accent-hover:#2980b9;--danger:#e74c3c;--muted:#666}
        body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;max-width:840px;margin:0 auto;padding:2rem;line-height:1.6;color:#333;background-color:#f9f9f9}
        .container{background:#fff;padding:3rem;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.08)}
        h1{color:var(--danger);margin:.25rem 0 1rem;font-size:2.25rem}
        .error-code{font-size:4rem;font-weight:800;color:var(--danger);margin:0;line-height:1}
        .requested-path{background:#f8f9fa;padding:.5rem 1rem;border-radius:6px;font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace;border-left:4px solid var(--danger);margin:1rem 0}
        .routes{background:#f8f9fa;padding:1rem;border-radius:6px;border-left:4px solid var(--accent);margin:1rem 0}
        .routes h3{margin:.25rem 0 0;color:var(--accent)}
        .routes p{margin:.35rem 0 1rem;color:#444}
        .routes-list{list-style:none;padding:0;margin:0}
        .route-item{display:flex;align-items:center;gap:.5rem;width:100%;text-align:left;background:transparent;border:1px solid transparent;border-radius:6px;padding:.5rem .75rem;font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace;color:#2c3e50;cursor:pointer}
        .route-item:focus{outline:3px solid rgba(52,152,219,.4)}
        .route-item:hover{background:#ecf0f1}
        .route-item .url{flex:1}
        .clipboard-icon{font-size:1.1em}
        .footer{margin-top:2rem;padding-top:1rem;border-top:1px solid #eee;color:var(--muted);font-size:.9rem}
        .back-link{display:inline-block;margin-top:1rem;padding:.75rem 1.25rem;background:var(--accent);color:#fff;text-decoration:none;border-radius:6px;transition:background .2s ease}
        .back-link:hover{background:var(--accent-hover)}
        @media(max-width:600px){body{padding:1rem}.container{padding:1.5rem}h1{font-size:1.75rem}.error-code{font-size:3rem}}
        `,
      ),
    );
  }

  // Optional custom message
  const customMessageSection = customMessage
    ? builder.createElement(
        'div',
        {
          role: 'note',
          'aria-label': 'Helpful note',
          style:
            'background:#fff3cd;padding:1rem;border-radius:6px;border-left:4px solid #ffc107;margin:1rem 0',
        },
        { escapeContent: false },
        builder.createElement('p', { style: 'margin:0' }, undefined, customMessage),
      )
    : '';

  // Routes listing (keyboard accessible, no inline handlers)
  const routesSection =
    availableRoutes.length > 0
      ? builder.createElement(
          'section',
          { class: 'routes', 'aria-labelledby': 'available-routes' },
          { escapeContent: false },
          builder.createElement('h3', { id: 'available-routes' }, undefined, '📍 Available Routes'),
          builder.createElement(
            'p',
            {},
            undefined,
            'Click or press Enter to copy a full URL to your clipboard.',
          ),
          builder.createElement(
            'ul',
            { class: 'routes-list' },
            { escapeContent: false },
            ...availableRoutes.map((route) =>
              builder.createElement(
                'li',
                {},
                { escapeContent: false },
                builder.createElement(
                  'button',
                  {
                    type: 'button',
                    class: 'route-item',
                    'data-route': route,
                    title: 'Copy route URL to clipboard',
                  },
                  { escapeContent: false },
                  builder.createElement('span', { class: 'url' }, undefined, route),
                  builder.createElement(
                    'span',
                    { class: 'clipboard-icon', 'aria-hidden': 'true' },
                    undefined,
                    '📋',
                  ),
                ),
              ),
            ),
          ),
        )
      : builder.createElement(
          'section',
          { class: 'routes' },
          { escapeContent: false },
          builder.createElement('h3', {}, undefined, '🔍 No Routes Available'),
          builder.createElement(
            'p',
            {},
            undefined,
            'No routes are currently registered on this server.',
          ),
        );

  // Main body (single element string to avoid fragment escaping)
  const body = builder.createElement(
    'div',
    { class: 'container', role: 'main' },
    { escapeContent: false },
    builder.createElement('div', { class: 'error-code', 'aria-hidden': 'true' }, undefined, '404'),
    builder.createElement('h1', {}, undefined, heading),
    builder.createElement(
      'p',
      {},
      undefined,
      'The requested page could not be found on this server.',
    ),
    builder.createElement(
      'div',
      { class: 'requested-path', 'aria-label': 'Requested path' },
      { escapeContent: false },
      builder.createElement('strong', {}, undefined, 'Requested:'),
      ' ' + requestedPath,
    ),
    customMessageSection,
    routesSection,
    builder.createElement('a', { href: '/', class: 'back-link' }, undefined, '← Go to Home'),
    builder.createElement(
      'div',
      { class: 'footer' },
      { escapeContent: false },
      builder.createElement(
        'p',
        {},
        { escapeContent: false },
        builder.createElement('strong', {}, undefined, serverName),
        ' • Generated at ' + new Date().toLocaleString(),
      ),
    ),
  );

  // Scripts: unobtrusive, delegated event handlers
  const scripts = builder.createElement(
    'script',
    {},
    undefined,
    `(() => {
      function showCopyNotification(message) {
        const notif = document.createElement('div');
        notif.textContent = message;
        Object.assign(notif.style, {
          position: 'fixed', bottom: '2rem', left: '50%', transform: 'translateX(-50%)',
          background: '#3498db', color: 'white', padding: '0.6rem 1rem', borderRadius: '6px',
          fontSize: '0.95rem', boxShadow: '0 2px 8px rgba(0,0,0,0.15)', zIndex: '9999', transition: 'opacity .25s ease'
        });
        document.body.appendChild(notif);
        setTimeout(() => { notif.style.opacity = '0'; setTimeout(() => notif.remove(), 250); }, 1200);
      }

      async function copyRoute(route) {
        const fullUrl = window.location.origin + route;
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(fullUrl);
          } else {
            const temp = document.createElement('input');
            temp.value = fullUrl; document.body.appendChild(temp); temp.select();
            document.execCommand('copy');
            document.body.removeChild(temp);
          }
          showCopyNotification('Copied: ' + fullUrl);
        } catch {
          showCopyNotification('Failed to copy');
        }
      }

      function handleActivate(el) {
        const route = el.getAttribute('data-route');
        if (route) copyRoute(route);
      }

      document.addEventListener('click', (e) => {
        const target = e.target;
        if (!(target instanceof Element)) return;
        const btn = target.closest('button.route-item');
        if (btn) handleActivate(btn);
      });

      document.addEventListener('keydown', (e) => {
        const target = e.target;
        if (!(target instanceof Element)) return;
        if (target.matches('button.route-item') && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          handleActivate(target);
        }
      });
    })();`,
  );

  // Compose full HTML document
  return PageBuilder.buildPage({
    title,
    head: String(head),
    body: String(body),
    scripts: String(scripts),
    lang: 'en',
    charset: 'utf-8',
  });
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
