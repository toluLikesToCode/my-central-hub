/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

import dayjs from 'dayjs';
import {
  formatDate,
  formatShortDate,
  formatTime,
  getCurrentFormattedDate,
  humanizeTimestamp,
} from '../../src/utils/dateFormatter';

// Mock dayjs and its extensions
jest.mock('dayjs', () => {
  const tzMock = {
    format: jest.fn((fmt) => {
      if (fmt === 'MMM DD, YYYY hh:mm:ss A z') {
        return 'May 04, 2025 01:56:21 PM PDT';
      } else if (fmt === 'MMM DD, YYYY') {
        return 'May 04, 2025';
      } else if (fmt === 'hh:mm:ss A z') {
        return '01:56:21 PM PDT';
      }
      return 'MOCK DATE FORMAT';
    }),
  };

  const mockDayjs = jest.fn(() => ({
    tz: jest.fn(() => tzMock),
  }));

  // Add extensions as functions
  mockDayjs.extend = jest.fn();
  mockDayjs.tz = {
    setDefault: jest.fn(),
  };

  return mockDayjs;
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
    expect(dayjs).toHaveBeenCalled();
  });

  test('formatShortDate should format date using short format', () => {
    const result = formatShortDate(testDate);
    expect(result).toBe('May 04, 2025');
    expect(dayjs).toHaveBeenCalled();
  });

  test('formatTime should format date with time only', () => {
    const result = formatTime(testDate);
    expect(result).toBe('01:56:21 PM PDT');
    expect(dayjs).toHaveBeenCalled();
  });

  test('getCurrentFormattedDate should get current date formatted', () => {
    const result = getCurrentFormattedDate();
    expect(result).toBe('May 04, 2025 01:56:21 PM PDT');
    expect(dayjs).toHaveBeenCalled();
  });

  test('humanizeTimestamp should convert ISO date to human readable format', () => {
    const result = humanizeTimestamp(testIsoString);
    expect(result).toBe('May 04, 2025 01:56:21 PM PDT');
    expect(dayjs).toHaveBeenCalled();
  });

  test('formatDate should handle date string input', () => {
    const result = formatDate(testIsoString);
    expect(result).toBe('May 04, 2025 01:56:21 PM PDT');
    expect(dayjs).toHaveBeenCalled();
  });

  test('formatDate should handle timestamp number input', () => {
    const timestamp = testDate.getTime();
    const result = formatDate(timestamp);
    expect(result).toBe('May 04, 2025 01:56:21 PM PDT');
    expect(dayjs).toHaveBeenCalled();
  });
});
