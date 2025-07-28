// filepath: /Users/toluadegbehingbe/my-central-hub/src/core/middlewares/optionsHandler.ts
import { Middleware } from '../router';
import { handlePreflightRequest } from './cors';
import logger from '../../utils/logger';

/**
 * Middleware for handling OPTIONS requests
 * This middleware intercepts OPTIONS requests and responds with appropriate CORS headers
 * before they reach any route handlers
 */
export const optionsHandlerMiddleware: Middleware = (req, sock, next) => {
  if (req.method === 'OPTIONS') {
    logger.debug('[optionsHandlerMiddleware] Intercepting OPTIONS request', {
      path: req.path,
      requestId: req.ctx?.requestId,
    });

    // Use the handlePreflightRequest function from CORS middleware
    handlePreflightRequest(req, sock);

    // Don't continue to other middlewares or route handlers
    return;
  }

  // Not an OPTIONS request, continue middleware chain
  return next();
};
