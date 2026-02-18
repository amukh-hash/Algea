# Feature Flags

Flags:
- `ui.shellV2`
- `ui.executionV2`
- `ui.researchV2`
- `ui.runDetailV2`
- `ui.compareV2`

Behavior:
- Local default: enabled.
- Prod default: disabled unless enabled by env.
- Query param override: `?ff=shellV2,executionV2`.
- Settings page shows active flags.
