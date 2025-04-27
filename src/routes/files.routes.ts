// routes/files.routes.ts
import { router } from '../core/router';
import { fileStreamingController } from '../modules/file-streamer';

router.get('/files', fileStreamingController.listFiles);
