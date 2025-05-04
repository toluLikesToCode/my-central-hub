/**
 * src/routes/embeddings.routes.ts
 * This file defines the routes for the embeddings service.
 */
import { config } from '../config/server.config';
import router from '../core/router';
import { embeddingsController } from '../modules/embeddings/embeddings.handler';

// Placeholder: Add feature flag check if needed, similar to other routes
// if (config.features.embeddings) {
if (config.features.embeddingService) {
  router.post('/api/embeddings', embeddingsController.handleEmbeddingsRequest);
  router.any('/api/embeddings/shutdown', embeddingsController.handleShutdownRequest);
  router.get('/api/embeddings/status', embeddingsController.handleStatusRequest);
}
