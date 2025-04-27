// routes/stream.routes.ts
import { router } from '../core/router';
import { fileStreamingController } from '../modules/file-streamer';

router.get('/stream', fileStreamingController.handleStream);
