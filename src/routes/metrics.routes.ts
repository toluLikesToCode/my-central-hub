// routes/metrics.routes.ts
import { router } from '../core/router';
import { metricsController } from '../modules/metrics/metricsController';

/**
 * Accepts the JSON payload, validates, and writes into SQLite.
 */
router.post('/metrics', metricsController.handleMetrics);
