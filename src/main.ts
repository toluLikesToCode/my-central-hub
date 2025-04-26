import { HttpServer } from "./core/server";
import { config } from "./config/server.config";
import { logger } from "./utils/logger";

const server = new HttpServer(config.port);
server.start();
