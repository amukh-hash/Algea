# UI Architecture (V2)

## Layers
1. **Theme tokens** in `src/theme/tokens/*.json` built into CSS variables via `src/theme/buildTokens.ts` and `src/theme/generated.css`.
2. **Design system primitives** in `src/components/ui/*`.
3. **App shell** in `src/layout/AppShell.tsx` with top bar, sidenav, breadcrumbs, command palette, and global toasts.
4. **Route pages** under `src/app/*` consuming primitives and React Query data.
5. **Realtime subsystem** in `src/realtime/*` with bounded windows and connection state.

## Realtime Memory Bounds
- Metric points per key: 200
- Chart points per key: 1000
- Events window: 200
