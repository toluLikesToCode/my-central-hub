# App Metrics Module

This module provides a structure for collecting and exposing metrics for different applications within the Central Hub project.

## Structure

- Each app has its own subfolder under `app-metrics/` (e.g., `app_gallery-generator/`).
- Each subfolder contains its own controller, service, and index.ts for modularity.
- The main `index.ts` re-exports all app metrics modules.

## Adding Metrics for a New App

1. **Create a new folder:**
   - Example: `src/modules/app-metrics/my_new_app/`
2. **Add your controller and service:**
   - `metricsController.ts` and `metricsService.ts`
3. **Export from `index.ts`:**
   - `export * from './metricsController';`
   - `export * from './metricsService';`
4. **Register the endpoint:**

   - In `src/routes/metrics.routes.ts`, add:

     ```typescript
     import { myNewAppMetricsController } from '../modules/app-metrics/my_new_app/metricsController';
     if (config.features.metrics) {
       router.post('/api/metrics/my-new-app', myNewAppMetricsController.handleMetrics);
     }
     ```

## Feature Toggling

- Metrics endpoints are only registered if `config.features.metrics` is enabled in `src/config/server.config.ts`.
- To disable all metrics endpoints, set `metrics: false` in the config.

## Example Folder Structure

```
src/modules/app-metrics/
  app_gallery-generator/
    metricsController.ts
    metricsService.ts
    index.ts
  my_new_app/
    metricsController.ts
    metricsService.ts
    index.ts
  index.ts
  README.md
```

## Best Practices

- Use RESTful route patterns: `/api/metrics/:app`.
- Keep each app’s metrics logic isolated in its own folder.
- Document new endpoints in the main README and provide usage examples.
