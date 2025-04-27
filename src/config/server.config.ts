import dotenv from 'dotenv';
import { join } from 'path';
import { logger } from '../utils/logger';

// Load environment variables from .env file
dotenv.config();

export const config = {
  port: process.env.PORT ? parseInt(process.env.PORT, 10) : 8080,
  publicDir: process.env.PUBLIC_DIR
    ? join(process.cwd(), process.env.PUBLIC_DIR)
    : join(process.cwd(), 'public'),
  mediaDir: process.env.MEDIA_DIR
    ? join(process.cwd(), process.env.MEDIA_DIR)
    : join(process.cwd(), 'media'),
};

logger.info(`Server configuration:`);
logger.info(`- Port: ${config.port}`);
logger.info(`- Public Directory: ${config.publicDir}`);
logger.info(`- Media Directory: ${config.mediaDir}`);
