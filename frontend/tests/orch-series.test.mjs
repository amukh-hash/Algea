import test from 'node:test';
import assert from 'node:assert/strict';

function parseSeries(series) {
  if (!Array.isArray(series)) throw new Error('series must be array');
  return series.map((p) => ({
    t: String(p.t),
    cum_net_unscaled: Number(p.cum_net_unscaled ?? 0),
    cum_net_volscaled: Number(p.cum_net_volscaled ?? 0),
  }));
}

test('equity-series parser keeps required keys', () => {
  const out = parseSeries([{ t: '2026-02-18', cum_net_unscaled: '0.1', cum_net_volscaled: '0.08' }]);
  assert.equal(out[0].t, '2026-02-18');
  assert.equal(out[0].cum_net_unscaled, 0.1);
  assert.equal(out[0].cum_net_volscaled, 0.08);
});

test('equity-series parser handles empty', () => {
  const out = parseSeries([]);
  assert.equal(out.length, 0);
});
