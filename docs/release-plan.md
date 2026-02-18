# Release Plan
1. Deploy with all `ui.*V2` flags OFF in production.
2. Enable `shellV2` and `executionV2` for internal users via query param canary.
3. Expand to `researchV2`, then `runDetailV2`, then `compareV2`.
4. Monitor reconnect/gap toasts and SSE connection counts.
5. Rollback by disabling flags.
