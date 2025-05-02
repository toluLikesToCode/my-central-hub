import router from '../core/router';
import { embeddingsController } from '../modules/embeddings/embeddings.handler';

// Placeholder: Add feature flag check if needed, similar to other routes
// if (config.features.embeddings) {
router.post('/api/embeddings', embeddingsController.handleEmbeddingsRequest);
router.post('/api/embeddings/shutdown', embeddingsController.handleShutdownRequest);
router.get('/api/embeddings/status', embeddingsController.handleStatusRequest);
