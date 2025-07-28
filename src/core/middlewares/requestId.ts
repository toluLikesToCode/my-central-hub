//// filepath: /Users/toluadegbehingbe/my-central-hub/src/core/middlewares/requestId.ts
import { Middleware } from '../router';
import logger from '../../utils/logger';

export const requestIdMiddleware: Middleware = (req, _sock, next) => {
  req.ctx ??= {};
  req.ctx.requestId = req.ctx.requestId || generateRequestId();
  logger.debug('[requestIdMiddleware] Request ID generated', {
    requestId: req.ctx.requestId,
    request: {
      req,
    },
  });
  return next();
};

/**
 * Generate a unique request ID for tracking requests in logs
 */
function generateRequestId(): string {
  return 'request' + Date.now().toString(36) + Math.random().toString(36).substring(2);
}
