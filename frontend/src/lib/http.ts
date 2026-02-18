export class HttpError extends Error {
  status?: number;
  body?: unknown;
  timeout?: boolean;
  constructor(message: string, init?: Partial<HttpError>) {
    super(message);
    Object.assign(this, init);
  }
}

export async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit = {}, timeoutMs = 15_000): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(input, { ...init, signal: controller.signal });
    return response;
  } catch (error) {
    if ((error as Error).name === "AbortError") {
      throw new HttpError("Request timed out", { timeout: true });
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

export async function fetchJsonWithTimeout<T>(url: string, init: RequestInit = {}, timeoutMs = 15_000): Promise<T> {
  const response = await fetchWithTimeout(url, init, timeoutMs);
  const contentType = response.headers.get("content-type") ?? "";
  const body = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    throw new HttpError(`HTTP ${response.status}`, { status: response.status, body });
  }
  return body as T;
}
