// filepath: /Users/toluadegbehingbe/my-central-hub/src/utils/dateFormatter.ts
/**
 * src/utils/dateFormatter.ts
 * Provides utilities for formatting dates in a human-readable format
 * using the server configuration for timezone and format patterns.
 */
import dayjs from 'dayjs';
import utc from 'dayjs/plugin/utc';
import timezone from 'dayjs/plugin/timezone';
import advancedFormat from 'dayjs/plugin/advancedFormat';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Configure dayjs with necessary plugins
dayjs.extend(utc);
dayjs.extend(timezone);
dayjs.extend(advancedFormat);

// Define date format constants - read from env vars or use defaults
const DEFAULT_TIMEZONE = process.env.TIMEZONE || 'America/Vancouver'; // PDT (Vancouver) timezone
const DEFAULT_DATE_FORMAT = process.env.DATE_FORMAT || 'MMM DD, YYYY hh:mm:ss A z'; // May 04, 2025 01:56:21 PM PDT
const DEFAULT_SHORT_DATE_FORMAT = process.env.SHORT_DATE_FORMAT || 'MMM DD, YYYY'; // May 04, 2025
const DEFAULT_TIME_FORMAT = process.env.TIME_FORMAT || 'hh:mm:ss A z'; // 01:56:21 PM PDT

// Set the default timezone
dayjs.tz.setDefault(DEFAULT_TIMEZONE);

/**
 * Formats a date or timestamp into a human-readable string
 * using the configured format and timezone
 *
 * @param date - Date to format (Date object, ISO string, timestamp, etc.)
 * @param format - Optional format string override
 * @returns Formatted date string
 */
export function formatDate(date: Date | string | number, format?: string): string {
  return dayjs(date)
    .tz(DEFAULT_TIMEZONE)
    .format(format || DEFAULT_DATE_FORMAT);
}

/**
 * Formats a date as a short date (without time)
 *
 * @param date - Date to format (Date object, ISO string, timestamp, etc.)
 * @returns Formatted short date string
 */
export function formatShortDate(date: Date | string | number): string {
  return dayjs(date).tz(DEFAULT_TIMEZONE).format(DEFAULT_SHORT_DATE_FORMAT);
}

/**
 * Formats a time only (without date)
 *
 * @param date - Date to format (Date object, ISO string, timestamp, etc.)
 * @returns Formatted time string
 */
export function formatTime(date: Date | string | number): string {
  return dayjs(date).tz(DEFAULT_TIMEZONE).format(DEFAULT_TIME_FORMAT);
}

/**
 * Gets the current date and time formatted according to config
 *
 * @param format - Optional format string override
 * @returns Current date formatted as string
 */
export function getCurrentFormattedDate(format?: string): string {
  return formatDate(new Date(), format);
}

/**
 * Transforms an ISO timestamp to a human-readable format
 *
 * @param isoString - ISO date string to transform
 * @returns Human-readable date string
 */
export function humanizeTimestamp(isoString: string): string {
  return formatDate(isoString);
}

// Export date format constants for usage in other modules
export const DateTimeConfig = {
  timezone: DEFAULT_TIMEZONE,
  format: DEFAULT_DATE_FORMAT,
  shortFormat: DEFAULT_SHORT_DATE_FORMAT,
  timeFormat: DEFAULT_TIME_FORMAT,
};
