export class HttpError extends Error {
  constructor(message, init = {}) { super(message); Object.assign(this, init); }
}

export async function fetchWithTimeout(input, init = {}, timeoutMs = 50) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } catch (error) {
    if (error.name === 'AbortError') throw new HttpError('Request timed out', { timeout: true });
    throw error;
  } finally { clearTimeout(timer); }
}
