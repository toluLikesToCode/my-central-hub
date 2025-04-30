// routes/files.routes.ts
// DEPRECATED: This file is now handled by file-hosting.routes.ts for all /api/files endpoints.
// Please use file-hosting.routes.ts instead.

import router from '../core/router';
import { fileHostingController } from '../modules/file-hosting';

router.get('/files', fileHostingController.listFiles);
