import test from 'node:test';
import assert from 'node:assert/strict';
import { shouldAcceptEvent, hasGap, boundMetric, boundChart, boundEvents } from '../src/realtime/utils.js';

test('accepts only increasing event ids', () => {
  assert.equal(shouldAcceptEvent(10, 11), true);
  assert.equal(shouldAcceptEvent(10, 10), false);
  assert.equal(shouldAcceptEvent(10, 9), false);
});

test('detects stream gap', () => {
  assert.equal(hasGap(0, 10), false);
  assert.equal(hasGap(10, 11), false);
  assert.equal(hasGap(10, 15), true);
});

test('enforces bounded windows', () => {
  const arr = Array.from({ length: 1200 }, (_, i) => i);
  assert.equal(boundMetric(arr).length, 200);
  assert.equal(boundChart(arr).length, 1000);
  assert.equal(boundEvents(arr).length, 200);
});
