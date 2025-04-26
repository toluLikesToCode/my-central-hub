import { createServer, Socket } from "net";
import { parser } from "./parser";
import { router } from "./router";
import { logger } from "../utils/logger";

export class HttpServer {
    private server = createServer();

    constructor(private readonly port: number) {
        this.setupServer();
    }

    private setupServer() {
        this.server.on("connection", (socket: Socket) => {
            logger.info("New connection established.");
            socket.on("data", async (chunk: Buffer) => {
                try {
                    const request = parser.parse(chunk.toString());
                    const response = await router.handle(request, socket);
                    // router handles response sending
                } catch (error) {
                    logger.error(`Failed to process request: ${(error as Error).message}`);
                    socket.end("HTTP/1.1 400 Bad Request\r\n\r\nBad Request");
                }
            });

            socket.on("error", (err) => {
                logger.error(`Socket error: ${err.message}`);
            });
        });

        this.server.on("error", (err) => {
            logger.error(`Server error: ${err.message}`);
        });
    }

    public start() {
        this.server.listen(this.port, () => {
            logger.info(`🚀 Server running at port ${this.port}`);
        });
    }
}
