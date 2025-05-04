import router from '../core/router';
import { fileHostingController } from '../modules/file-hosting';
import { config } from '../config/server.config';

if (config.features.fileHosting) {
  // List all files
  router.get('/api/files', fileHostingController.listFiles);
  // Upload a file
  router.post('/api/files', fileHostingController.uploadFile);
  // Get a specific file
  router.get('/api/files/:filename', fileHostingController.getFile);
  // Delete a specific file
  router.del('/api/files/:filename', fileHostingController.deleteFile);
}
