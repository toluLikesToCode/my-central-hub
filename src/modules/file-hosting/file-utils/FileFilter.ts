import { FileStats } from '../fileHostingStatsHelper';
import getLoggerInstance from './fileHostingLogger';

const log = getLoggerInstance({ context: 'fileFilter' });

/**
 * A flexible, composable filter description understood by `evaluateFilter`.
 * All criteria are optional and may be freely combined.
 */
export interface FileFilter {
  /* Boolean logic ------------------------------------------------------- */
  not?: FileFilter; // Negation ( !A )
  and?: FileFilter[]; // Conjunction ( A ∧ B ∧ … )
  or?: FileFilter[]; // Disjunction ( A ∨ B ∨ … )

  /* Pattern matching ---------------------------------------------------- */
  regex?: Record<string, string>; // `{ propertyName: "pattern" }`

  /* Simple equality / range clauses ------------------------------------ */
  fileName?: string;
  mimeType?: string;
  extension?: string; // e.g. "mp4"
  minSize?: number;
  maxSize?: number;
  minWidth?: number;
  minHeight?: number;
  minDuration?: number; // seconds
  dateFrom?: string | Date; // inclusive
  dateTo?: string | Date; // inclusive
}

/**
 * Advanced filter evaluator for {@link FileStats} records.
 * Every field in {@link FileFilter} is optional; unspecified fields are ignored.
 */
export function evaluateFilter(stats: FileStats, filter: FileFilter = {}): boolean {
  /* ---------- Boolean logic ---------- */
  // If 'and' is present, all subfilters must match; if any fails, return false
  if (filter.and && !filter.and.every((sub) => evaluateFilter(stats, sub))) {
    log.debug('AND filter failed', {
      fileName: stats.fileName,
      mimeType: stats.mimeType,
      filter: filter.and,
    });
    return false;
  }
  // If 'or' is present, at least one subfilter must match; if none match, return false
  if (filter.or && !filter.or.some((sub) => evaluateFilter(stats, sub))) {
    log.debug('OR filter failed', {
      fileName: stats.fileName,
      mimeType: stats.mimeType,
      filter: filter.or,
    });
    return false;
  }

  /* ---------- Regex matching ---------- */
  // If 'regex' is present, each property must match its regex pattern; if any fails, return false
  if (filter.regex) {
    for (const [prop, pattern] of Object.entries(filter.regex)) {
      const value = (stats as unknown as Record<string, unknown>)[prop];
      // If the property is not a string or doesn't match the pattern, return false
      if (typeof value !== 'string' || !new RegExp(pattern).test(value)) {
        log.debug('Regex filter failed', {
          fileName: stats.fileName,
          mimeType: stats.mimeType,
          prop,
          pattern,
          value,
        });
        return false;
      }
    }
  }

  /* ---------- Scalar / range checks ---------- */
  // If 'fileName' is present, file name must include the filter value; otherwise, return false
  if (filter.fileName && !stats.fileName?.includes(filter.fileName)) {
    log.debug('File name does not match filter:', {
      fileName: stats.fileName,
      mimeType: stats.mimeType,
      filter: filter.fileName,
    });
    return false;
  }

  // If 'mimeType' is present, mime type must include the filter value; otherwise, return false
  if (filter.mimeType && !stats.mimeType?.includes(filter.mimeType)) {
    log.debug('MIME type does not match filter:', {
      fileName: stats.fileName,
      mimeType: stats.mimeType,
      filter: filter.mimeType,
    });
    return false;
  }

  // If 'minSize' is present, file size must be at least minSize; otherwise, return false
  if (filter.minSize !== undefined && stats.size < filter.minSize) {
    log.debug('minSize filter failed', {
      fileName: stats.fileName,
      mimeType: stats.mimeType,
      minSize: filter.minSize,
      size: stats.size,
    });
    return false;
  }
  // If 'maxSize' is present, file size must be at most maxSize; otherwise, return false
  if (filter.maxSize !== undefined && stats.size > filter.maxSize) {
    log.debug('maxSize filter failed', {
      fileName: stats.fileName,
      mimeType: stats.mimeType,
      maxSize: filter.maxSize,
      size: stats.size,
    });
    return false;
  }

  // If 'dateFrom' is present, lastModified must be on or after dateFrom; otherwise, return false
  if (filter.dateFrom) {
    const from = filter.dateFrom instanceof Date ? filter.dateFrom : new Date(filter.dateFrom);
    if (stats.lastModified < from) {
      log.debug('dateFrom filter failed', {
        fileName: stats.fileName,
        mimeType: stats.mimeType,
        dateFrom: filter.dateFrom,
        lastModified: stats.lastModified,
      });
      return false;
    }
  }
  // If 'dateTo' is present, lastModified must be on or before dateTo; otherwise, return false
  if (filter.dateTo) {
    const to = filter.dateTo instanceof Date ? filter.dateTo : new Date(filter.dateTo);
    if (stats.lastModified > to) {
      log.debug('dateTo filter failed', {
        fileName: stats.fileName,
        mimeType: stats.mimeType,
        dateTo: filter.dateTo,
        lastModified: stats.lastModified,
      });
      return false;
    }
  }

  // If 'minWidth' is present, width must be at least minWidth; otherwise, return false
  if (filter.minWidth !== undefined && (stats.width ?? 0) < filter.minWidth) {
    log.debug('minWidth filter failed', {
      fileName: stats.fileName,
      mimeType: stats.mimeType,
      minWidth: filter.minWidth,
      width: stats.width,
    });
    return false;
  }
  // If 'minHeight' is present, height must be at least minHeight; otherwise, return false
  if (filter.minHeight !== undefined && (stats.height ?? 0) < filter.minHeight) {
    log.debug('minHeight filter failed', {
      fileName: stats.fileName,
      mimeType: stats.mimeType,
      minHeight: filter.minHeight,
      height: stats.height,
    });
    return false;
  }
  // If 'minDuration' is present, duration must be at least minDuration; otherwise, return false
  if (filter.minDuration !== undefined && (stats.duration ?? 0) < filter.minDuration) {
    log.debug('minDuration filter failed', {
      fileName: stats.fileName,
      mimeType: stats.mimeType,
      minDuration: filter.minDuration,
      duration: stats.duration,
    });
    return false;
  }

  // If 'extension' is present, file extension must match (case-insensitive); otherwise, return false
  if (filter.extension) {
    const ext = stats.fileName?.split('.').pop()?.toLowerCase() ?? '';
    if (ext !== filter.extension.toLowerCase()) {
      log.debug('extension filter failed', {
        fileName: stats.fileName,
        mimeType: stats.mimeType,
        extension: filter.extension,
        ext,
      });
      return false;
    }
  }

  /* ---------- NOT filter (must be last) ---------- */
  // If 'not' is present and its subfilter matches, return false (negation)
  if (filter.not && typeof filter.not === 'object') {
    if (evaluateFilter(stats, filter.not)) {
      log.debug('Filter negation failed:', {
        fileName: stats.fileName,
        mimeType: stats.mimeType,
        filter: filter.not,
      });
      return false;
    }
  }

  return true;
}
