import { IncomingRequest } from "../entities/http";
import { logger } from "../utils/logger";

export const parser = {
    parse(rawData: string): IncomingRequest {
        try {
            const [headerPart] = rawData.split("\r\n\r\n");
            const [requestLine, ...headers] = headerPart.split("\r\n");
            const [method, path, httpVersion] = requestLine.split(" ");

            if (!method || !path || !httpVersion) {
                throw new Error("Malformed request line");
            }

            return {
                method,
                path,
                httpVersion,
                headers: headers.reduce((acc, curr) => {
                    const [key, value] = curr.split(": ");
                    acc[key.toLowerCase()] = value;
                    return acc;
                }, {} as Record<string, string>),
                raw: rawData,
            };
        } catch (error) {
            logger.error(`Failed to parse request: ${(error as Error).message}`);
            return {
                method: undefined,
                path: undefined,
                httpVersion: undefined,
                headers: {},
                raw: rawData,
            };
        }
    },
};
