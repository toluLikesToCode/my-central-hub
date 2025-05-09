import { FileStats } from '../fileHostingStatsHelper';
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
  if (filter.not && evaluateFilter(stats, filter.not)) return false;

  if (filter.and && !filter.and.every((sub) => evaluateFilter(stats, sub))) return false;

  if (filter.or && !filter.or.some((sub) => evaluateFilter(stats, sub))) return false;

  /* ---------- Regex matching ---------- */
  if (filter.regex) {
    for (const [prop, pattern] of Object.entries(filter.regex)) {
      const value = (stats as unknown as Record<string, unknown>)[prop];
      if (typeof value !== 'string' || !new RegExp(pattern).test(value)) return false;
    }
  }

  /* ---------- Scalar / range checks ---------- */
  if (filter.fileName && !stats.fileName.includes(filter.fileName)) return false;
  if (filter.mimeType && !stats.mimeType.includes(filter.mimeType)) return false;

  if (filter.minSize !== undefined && stats.size < filter.minSize) return false;
  if (filter.maxSize !== undefined && stats.size > filter.maxSize) return false;

  if (filter.dateFrom) {
    const from = filter.dateFrom instanceof Date ? filter.dateFrom : new Date(filter.dateFrom);
    if (stats.lastModified < from) return false;
  }
  if (filter.dateTo) {
    const to = filter.dateTo instanceof Date ? filter.dateTo : new Date(filter.dateTo);
    if (stats.lastModified > to) return false;
  }

  if (filter.minWidth !== undefined && (stats.width ?? 0) < filter.minWidth) return false;
  if (filter.minHeight !== undefined && (stats.height ?? 0) < filter.minHeight) return false;
  if (filter.minDuration !== undefined && (stats.duration ?? 0) < filter.minDuration) return false;

  if (filter.extension) {
    const ext = stats.fileName.split('.').pop()?.toLowerCase() ?? '';
    if (ext !== filter.extension.toLowerCase()) return false;
  }

  return true;
}
