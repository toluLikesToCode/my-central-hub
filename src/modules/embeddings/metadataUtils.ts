/**
 * Utility functions for mapping between Python snake_case and TypeScript camelCase fields
 */

/**
 * Convert a snake_case string to camelCase
 * @param snakeStr String in snake_case format
 * @returns The converted camelCase string
 */
export function snakeToCamel(snakeStr: string): string {
  return snakeStr.replace(/([-_][a-z])/g, (group) =>
    group.toUpperCase().replace('-', '').replace('_', ''),
  );
}

/**
 * Recursively transforms an object with snake_case keys to camelCase keys
 * @param data Object with snake_case keys
 * @returns New object with camelCase keys
 */
export function transformKeysToCamelCase<T extends object>(data: unknown): T {
  if (Array.isArray(data)) {
    return data.map((item) =>
      typeof item === 'object' && item !== null ? transformKeysToCamelCase(item) : item,
    ) as unknown as T;
  }

  if (typeof data !== 'object' || data === null) {
    return data as T;
  }

  const dataObj = data as Record<string, unknown>;

  return Object.keys(dataObj).reduce(
    (result, key) => {
      const camelKey = snakeToCamel(key);
      const value = dataObj[key];

      result[camelKey] =
        typeof value === 'object' && value !== null ? transformKeysToCamelCase(value) : value;

      return result;
    },
    {} as Record<string, unknown>,
  ) as T;
}

/**
 * Maps Python-specific metadata fields to TypeScript fields
 * @param metadata The debug metadata from Python embedding service
 * @returns Mapped and transformed metadata
 */
export function mapPythonMetadata(
  metadata: Record<string, unknown> | undefined,
): Record<string, unknown> {
  if (!metadata) return {};

  // First transform all keys to camelCase
  const camelCaseMetadata = transformKeysToCamelCase<Record<string, unknown>>(metadata);

  // Then handle any special case mappings that aren't just simple camelCase transformations
  const specialMappings: Record<string, string> = {
    // Add any special mappings here if the camelCase transformation isn't sufficient
    // Example: 'some_complex_name': 'simpleName'
  };

  // Apply any special mappings
  Object.entries(specialMappings).forEach(([originalKey, mappedKey]) => {
    if (originalKey in metadata && !(mappedKey in camelCaseMetadata)) {
      camelCaseMetadata[mappedKey] = metadata[originalKey];
    }
  });

  return camelCaseMetadata;
}
