// main.ts
import './routes';
import { HttpServer } from './core/server';
import { config } from './config/server.config';
import logger from './utils/logger';

logger.child([JSON.stringify(config, null, 2)]).info('routes loaded');

const server = new HttpServer(config.port);
server.start();
