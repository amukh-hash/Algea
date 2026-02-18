import test from 'node:test';
import assert from 'node:assert/strict';
import { fetchWithTimeout } from '../src/lib/http.testable.js';
import { resolveWebBase } from '../src/runtime/config.testable.js';

test('resolveWebBase defaults correctly', () => {
  assert.equal(resolveWebBase(undefined), 'http://localhost:8000');
  assert.equal(resolveWebBase('http://x:1'), 'http://x:1');
});

test('fetchWithTimeout marks timeout', async () => {
  const oldFetch = global.fetch;
  global.fetch = async (_input, init = {}) => {
    await new Promise((resolve, reject) => {
      init.signal?.addEventListener('abort', () => reject(Object.assign(new Error('Aborted'), { name: 'AbortError' })));
      setTimeout(resolve, 50);
    });
    return new Response('{}', { status: 200 });
  };

  try {
    await assert.rejects(
      () => fetchWithTimeout('http://example.com', {}, 1),
      (err) => Boolean(err.timeout),
    );
  } finally {
    global.fetch = oldFetch;
  }
});
