import { test } from 'node:test';
import assert from 'node:assert';
import { globalToastBus } from '../src/components/ui/ToastProvider.js';

test('globalToastBus can receive toasts without throwing', () => {
    // Before React mounts, it's a no-op
    assert.doesNotThrow(() => {
        globalToastBus.addToast({
            type: "info",
            title: "Test",
            description: "Testing global bus"
        });
    });
});

test('globalToastBus can be overridden by ToastProvider', () => {
    let received = null;

    // Mock the provider overriding the bus
    globalToastBus.addToast = (toast) => {
        received = toast;
    };

    globalToastBus.addToast({
        type: "error",
        title: "Timeout",
        description: "504 Gateway"
    });

    assert.notStrictEqual(received, null);
    assert.strictEqual(received.type, "error");
    assert.strictEqual(received.title, "Timeout");

    // Reset
    globalToastBus.addToast = () => { };
});
