# Feature Flags

Flags (query param `?ff=` overrides):
- `ui.shellV2` (`shellV2`)
- `ui.executionV2` (`executionV2`)
- `ui.researchV2` (`researchV2`)
- `ui.runDetailV2` (`runDetailV2`)
- `ui.compareV2` (`compareV2`)

Defaults:
- Local/non-prod: enabled
- Prod: disabled unless query override

Desktop mode note:
- In Tauri mode, runtime API base URL comes from backend supervisor command `get_backend_base_url`.
