import { HttpServer } from "./core/server";
import { config } from "./config/server.config";

const server = new HttpServer(config.port);
server.start();
