// routes/metrics.routes.ts
import router from '../core/router';
import { metricsController } from '../modules/app-metrics/app_gallery-generator';
import { config } from '../config/server.config';

/**
 * Accepts the JSON payload, validates, and writes into SQLite.
 */
if (config.features.metrics) {
  router.post('/api/metrics/gallery-generator', metricsController.handleMetrics);
}
