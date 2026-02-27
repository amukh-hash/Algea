# Migration Map

## Script Entrypoints
| Legacy Script | New Script |
| --- | --- |
| `backend/scripts/01_backfill_canonical_daily.py` | `backend/scripts/canonicalize/canonicalize_daily.py` |
| `backend/scripts/02_build_universe_manifests.py` | `backend/scripts/eligibility/build_eligibility.py` |
| `backend/scripts/03_build_featureframe.py` | `backend/scripts/features/build_features.py` |
| `backend/scripts/teacher/phase1_train_teacher_gold.py` | `backend/scripts/foundation/train_foundation.py` |
| `backend/scripts/teacher/build_priors_frame.py` | `backend/scripts/priors/build_priors.py` |
| `backend/scripts/selector/phase3_train_selector.py` | `backend/scripts/ranker/train_ranker.py` |
| `backend/scripts/07_nightly_run.py` | `backend/scripts/run/run_nightly_cycle.py` |
| (new) | `backend/scripts/research/run_backtest.py` |
| (new) | `backend/scripts/paper/run_paper_cycle.py` |
| (new) | `backend/scripts/live/run_live_cycle.py` |

## Artifact Paths
| Legacy Path | New Path |
| --- | --- |
| `backend/data_canonical/ohlcv_adj/` | `backend/artifacts/canonical/daily/` |
| `backend/manifests/` | `backend/artifacts/eligibility/` |
| `backend/features/` | `backend/artifacts/features/` |
| `backend/priors/` | `backend/artifacts/priors/` |
| `backend/outputs/` | `backend/artifacts/signals/` |
| `backend/reports/` | `backend/artifacts/reports/` |

## Module Imports
| Legacy Import | New Import |
| --- | --- |
| `backend.app.data.*` | `algea.data.*` |
| `backend.app.models.*` | `algea.models.*` |
| `backend.app.engine.*` | `algea.execution.*` |
| `backend.app.ops.*` | `algea.core.*` |

## Deprecated Code
Legacy code has been moved into `deprecated/backend_app_snapshot/` and `deprecated/legacy_scripts/`.
