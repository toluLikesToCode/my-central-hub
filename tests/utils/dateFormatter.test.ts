/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

import * as dayjs from 'dayjs';
import {
  formatDate,
  formatShortDate,
  formatTime,
  getCurrentFormattedDate,
  humanizeTimestamp,
} from '../../src/utils/dateFormatter';
import { config } from '../../src/config/server.config';

// Mock the dayjs timezone function
jest.mock('dayjs', () => {
  const originalDayjs = jest.requireActual('dayjs');
  const tzMock = {
    format: jest.fn((fmt) => {
      if (fmt === config.dateTime.format) {
        return 'May 04, 2025 01:56:21 PM PDT';
      } else if (fmt === config.dateTime.shortFormat) {
        return 'May 04, 2025';
      } else if (fmt === config.dateTime.timeFormat) {
        return '01:56:21 PM PDT';
      }
      return 'MOCK DATE FORMAT';
    }),
    tz: jest.fn(() => tzMock),
  };
  // Mock dayjs.extend to handle utc extension
  originalDayjs.extend = jest.fn();
  // Ensure dayjs.tz.setDefault exists at the top level and on default
  const tz = { setDefault: jest.fn() };
  const defaultExport = Object.assign(
    jest.fn(() => ({
      tz: jest.fn(() => tzMock),
    })),
    {
      extend: jest.fn(),
      tz,
    },
  );
  return {
    ...originalDayjs,
    __esModule: true,
    default: defaultExport,
    extend: jest.fn(),
    tz, // <-- ensure tz.setDefault exists
  };
});

describe('dateFormatter', () => {
  const testDate = new Date('2025-05-04T13:56:21.990Z');
  const testIsoString = '2025-05-04T13:56:21.990Z';

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('formatDate should format date using configured format', () => {
    const result = formatDate(testDate);
    expect(result).toBe('May 04, 2025 01:56:21 PM PDT');
    expect(dayjs.default).toHaveBeenCalled();
  });

  test('formatShortDate should format date using short format', () => {
    const result = formatShortDate(testDate);
    expect(result).toBe('May 04, 2025');
    expect(dayjs.default).toHaveBeenCalled();
  });

  test('formatTime should format date with time only', () => {
    const result = formatTime(testDate);
    expect(result).toBe('01:56:21 PM PDT');
    expect(dayjs.default).toHaveBeenCalled();
  });

  test('getCurrentFormattedDate should get current date formatted', () => {
    const result = getCurrentFormattedDate();
    expect(result).toBe('May 04, 2025 01:56:21 PM PDT');
    expect(dayjs.default).toHaveBeenCalled();
  });

  test('humanizeTimestamp should convert ISO date to human readable format', () => {
    const result = humanizeTimestamp(testIsoString);
    expect(result).toBe('May 04, 2025 01:56:21 PM PDT');
    expect(dayjs.default).toHaveBeenCalled();
  });

  test('formatDate should handle date string input', () => {
    const result = formatDate(testIsoString);
    expect(result).toBe('May 04, 2025 01:56:21 PM PDT');
    expect(dayjs.default).toHaveBeenCalled();
  });

  test('formatDate should handle timestamp number input', () => {
    const timestamp = testDate.getTime();
    const result = formatDate(timestamp);
    expect(result).toBe('May 04, 2025 01:56:21 PM PDT');
    expect(dayjs.default).toHaveBeenCalled();
  });
});
