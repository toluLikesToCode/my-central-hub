// routes/stream.routes.ts
import router from '../core/router';
import { fileStreamingController } from '../modules/file-streaming';
import { config } from '../config/server.config';

if (config.features.fileStreaming) {
  router.get('/api/stream', fileStreamingController.handleStream);
}
