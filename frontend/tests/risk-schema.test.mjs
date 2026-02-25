import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeRiskChecks } from '../src/lib/risk_schema.testable.js';

test('accepts canonical schema_version', () => {
  const out = normalizeRiskChecks({ schema_version: 'canonical', status: 'ok', violations: [], missing_sleeves: [], inputs: {}, metrics: {}, limits: {} });
  assert.equal(out.schema_version, 'canonical');
});

test('accepts legacy_normalized schema_version', () => {
  const out = normalizeRiskChecks({ schema_version: 'legacy_normalized', status: 'failed', violations: [{ code: 'X', message: 'x', details: {} }], missing_sleeves: [], inputs: {}, metrics: {}, limits: {} });
  assert.equal(out.schema_version, 'legacy_normalized');
});
