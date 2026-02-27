# ML Platform Config

## `tsfm_downsample_freq`

Controls **time-aware Chronos/TSFM downsampling frequency** before sliding-window construction.

- Type: duration string
- Default: `"1min"`
- Env vars:
  - Preferred: `TSFM_DOWNSAMPLE_FREQ_DURATION`
  - Deprecated: `TSFM_DOWNSAMPLE_FREQ` (warning emitted; value must still be a duration string)

### Allowed formats

- `<integer>min` (examples: `1min`, `5min`, `15min`)
- `<integer>h` (examples: `1h`, `2h`)
- `<integer>s` (examples: `30s`, `10s`)

### Behavior

Chronos preprocessing performs time-aware binning aligned to the floor of the first timestamp,
then takes the **last value in each bin** deterministically.

Example:

- Source: 1-minute bars from `09:30` to `09:39`
- `tsfm_downsample_freq="5min"`
- Downsampled timestamps: `09:30`, `09:35`
- Kept values: last value in `[09:30,09:35)` and last value in `[09:35,09:40)`

This replaces ambiguous integer-stride downsampling and ensures reproducible chronological behavior.
