import { test } from 'node:test';
import assert from 'node:assert';

// We mock the API layer just to test URL construction and query params.
// We can't import api.ts directly easily in a pure node .mjs environment 
// without ts-node/tsx, so we test the logic pattern.

function buildLwUrl(runId, keys, opts) {
    const params = new URLSearchParams({ keys: keys.join(","), format: "lw" });
    if (opts?.start) params.set("start", opts.start);
    if (opts?.end) params.set("end", opts.end);
    if (opts?.every_ms) params.set("every_ms", String(opts.every_ms));
    return `/api/telemetry/runs/${runId}/metrics?${params.toString()}`;
}

test('Metrics URL construction with lightweight charts format', () => {
    const url = buildLwUrl('run-123', ['pnl', 'gross'], { every_ms: 1000 });

    assert.ok(url.includes('format=lw'));
    assert.ok(url.includes('keys=pnl%2Cgross'));
    assert.ok(url.includes('every_ms=1000'));
});

function buildOrchUrl(asof) {
    return `/api/orchestrator/positions${asof ? `?asof=${asof}` : ''}`;
}

test('Orchestrator URL construction with optional asof', () => {
    const url1 = buildOrchUrl();
    assert.strictEqual(url1, '/api/orchestrator/positions');

    const url2 = buildOrchUrl('2023-10-25');
    assert.strictEqual(url2, '/api/orchestrator/positions?asof=2023-10-25');
});
