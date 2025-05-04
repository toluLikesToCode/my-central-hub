/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

import { PrettyFormatter, JsonFormatter } from '../../src/utils/logger';
import { formatDate } from '../../src/utils/dateFormatter';

// Mock the dateFormatter module
jest.mock('../../src/utils/dateFormatter', () => ({
  formatDate: jest.fn((date) => 'May 04, 2025 01:56:21 PM PDT'),
}));

describe('Logger Formatters with human-readable timestamps', () => {
  const testDate = new Date('2025-05-04T13:56:21.990Z');

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('JsonFormatter', () => {
    test('should format timestamps using human-readable format', () => {
      const formatter = new JsonFormatter();
      const entry = {
        level: 'info',
        message: 'Test message',
        timestamp: testDate,
        meta: { test: 'data' },
      };

      const formatted = formatter.format(entry);
      expect(formatDate).toHaveBeenCalledWith(testDate);
      expect(formatted).toContain('May 04, 2025 01:56:21 PM PDT');
      expect(formatted).not.toContain(testDate.toISOString());
    });

    test('should handle error formatting with human-readable timestamp', () => {
      const formatter = new JsonFormatter();
      const circularObj: never = {};
      circularObj.self = circularObj; // Create circular reference

      const entry = {
        level: 'error',
        message: circularObj,
        timestamp: testDate,
      };

      const formatted = formatter.format(entry);
      expect(formatDate).toHaveBeenCalledWith(testDate);
      expect(formatted).toContain('May 04, 2025 01:56:21 PM PDT');
      expect(formatted).toContain('Unserializable Object');
    });
  });

  describe('PrettyFormatter', () => {
    test('should format Date objects using human-readable format', () => {
      const formatter = new PrettyFormatter();
      const formatted = formatter['formatValue'](testDate);

      expect(formatDate).toHaveBeenCalledWith(testDate);
      expect(formatted).toContain('May 04, 2025 01:56:21 PM PDT');
      expect(formatted).not.toContain(testDate.toISOString());
    });

    test('should include human-readable timestamp in log entries when showTimestamp is true', () => {
      const formatter = new PrettyFormatter({ showTimestamp: true });
      const entry = {
        level: 'info',
        message: 'Test message',
        timestamp: testDate,
      };

      const formatted = formatter.format(entry);
      expect(formatDate).toHaveBeenCalledWith(testDate);
      expect(formatted).toContain('May 04, 2025 01:56:21 PM PDT');
      expect(formatted).not.toContain(testDate.toISOString());
    });
  });
});
