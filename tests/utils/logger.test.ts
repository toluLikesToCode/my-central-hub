import { logger } from "../../src/utils/logger";

describe("Logger", () => {
    it("should log info messages", () => {
        const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
        logger.info("Hello World");
        expect(consoleSpy).toHaveBeenCalled();
        consoleSpy.mockRestore();
    });

    it("should log warning messages", () => {
        const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
        logger.warn("Watch out!");
        expect(consoleSpy).toHaveBeenCalled();
        consoleSpy.mockRestore();
    });

    it("should log error messages", () => {
        const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
        logger.error("This is bad!");
        expect(consoleSpy).toHaveBeenCalled();
        consoleSpy.mockRestore();
    });

    it("should conditionally log debug messages", () => {
        process.env.NODE_ENV = "development";
        const consoleSpy = jest.spyOn(console, 'debug').mockImplementation();
        logger.debug("Debugging...");
        expect(consoleSpy).toHaveBeenCalled();
        consoleSpy.mockRestore();
    });
});
