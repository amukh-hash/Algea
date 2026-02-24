import { test } from 'node:test';
import assert from 'node:assert';

const staticCommands = [
    { href: "/execution", label: "Go to Execution" },
    { href: "/orchestrator", label: "Go to Orchestrator" },
    { href: "/research", label: "Go to Research" },
    { href: "/compare", label: "Go to Compare" },
    { href: "/settings", label: "Go to Settings" },
    { href: "/portfolio", label: "Go to Portfolio" },
];

function filterCommands(query) {
    let filtered = staticCommands;
    if (query) {
        filtered = filtered.filter(c => c.label.toLowerCase().includes(query.toLowerCase()));
        // If a user types a run UUID or typical UUID length, suggest jumping to it
        if (query.length > 8 && query.includes("-")) {
            filtered.push({ href: `/runs/${query}`, label: `View Run: ${query}` });
        }
    }
    return filtered;
}

test('Command palette filters static commands by query', () => {
    const res = filterCommands("port");
    assert.strictEqual(res.length, 1);
    assert.strictEqual(res[0].href, "/portfolio");
});

test('Command palette suggests run UUID navigation', () => {
    const runId = "123e4567-e89b-12d3";
    const res = filterCommands(runId);

    // It shouldn't match any static navs
    assert.strictEqual(res.length, 1);
    assert.strictEqual(res[0].href, `/runs/${runId}`);
    assert.strictEqual(res[0].label, `View Run: ${runId}`);
});

test('Command palette returns all when empty', () => {
    const res = filterCommands("");
    assert.strictEqual(res.length, 6);
});
