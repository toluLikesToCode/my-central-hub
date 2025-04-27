// main.ts
import './routes';
import { HttpServer } from './core/server';
import { config } from './config/server.config';
import { logger } from './utils/logger';
logger.info('routes loaded');

const server = new HttpServer(config.port);
server.start();
