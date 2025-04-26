import dotenv from "dotenv";
import { join } from "path";

// Load environment variables from .env file
dotenv.config();

export const config = {
    port: process.env.PORT ? parseInt(process.env.PORT, 10) : 8080,
    publicDir: process.env.PUBLIC_DIR ? join(process.cwd(), process.env.PUBLIC_DIR) : join(process.cwd(), "public"),
};
