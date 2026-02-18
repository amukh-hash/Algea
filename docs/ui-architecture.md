# UI Architecture

The frontend now uses a global `AppShell` with persistent top bar, side navigation, breadcrumbs, command palette (`Ctrl/Cmd+K`), and global realtime indicator.

## Layers
- Theme tokens (`frontend/src/theme/tokens/*`) -> generated CSS vars (`frontend/src/theme/generated.css`) -> Tailwind semantic colors.
- Design primitives (`frontend/src/components/ui/primitives.tsx`) shared across routes.
- Route pages (`/execution`, `/research`, `/runs/[runId]`, `/compare`) consume shared primitives.
- Realtime hooks in `frontend/src/realtime/*` separate transport (`useEventSource`) and run aggregation (`useRunStream`).
