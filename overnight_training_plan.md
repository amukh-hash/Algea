# Overnight Training Plan
**Created**: 2026-02-05 01:50 AM
**Estimated Completion**: 2026-02-05 ~7-8 AM

## Configuration
- **Total Steps**: 10,000 (2 runs × 5000 each)
- **Checkpoints**: 20 (every 500 steps)
- **Batch Size**: 4
- **Gradient Accumulation**: 8 (effective batch: 32)
- **Context Length**: 1024 tokens
- **Files**: 1000 (40% of dataset)
- **Learning Rate**: 5e-5

## Run 1: Epoch 1
- **Status**: Running
- **Steps**: 0-5000
- **Output**: `backend/models/teacher_e/gold_full/`
- **Checkpoints**: Every 500 steps (10 total)
- **ETA**: ~2-3 hours

## Run 2: Epoch 2
- **Status**: Queued (will start after Run 1)
- **Steps**: 5000-10000
- **Output**: `backend/models/teacher_e/gold_full_epoch2/`
- **Resume From**: `backend/models/teacher_e/gold_full/` (final checkpoint)
- **Checkpoints**: Every 500 steps (10 more)
- **ETA**: ~2-3 hours

## Expected Results
By morning, you will have:
- 20 checkpoints showing loss progression
- Full learning curve from random init to 10K steps
- Can compare checkpoints to find optimal stopping point
- Validation data to assess if model is learning (loss should decrease from 8.32)

## Files to Check in Morning
1. `backend/models/teacher_e/gold_full/run_manifest.json` - Epoch 1 final loss
2. `backend/models/teacher_e/gold_full_epoch2/run_manifest.json` - Epoch 2 final loss
3. Look for checkpoint directories: `checkpoint-500`, `checkpoint-1000`, etc.
4. Training logs (if any crashes occurred)

## Success Criteria
- Loss decreases below 7.0 (better than random)
- Loss continues decreasing through both epochs
- All 20 checkpoints saved successfully
