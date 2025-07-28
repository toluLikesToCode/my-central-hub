// Add explicit access to getRecentSessions and getSessionPerformanceStats
import { metricsController } from './metricsController';
import {
  saveMetrics,
  isPerfLogArray,
  getRecentSessions,
  getSessionPerformanceStats,
} from './metricsService';

export {
  metricsController,
  saveMetrics,
  isPerfLogArray,
  getRecentSessions,
  getSessionPerformanceStats,
};
