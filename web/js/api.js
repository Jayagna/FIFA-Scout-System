// Thin client for the JSON API in server.py.

export class ApiError extends Error {
  constructor(status, body) {
    super(body?.message || `Request failed (${status})`);
    this.status = status;
    this.code = body?.error;
  }
}

const cache = new Map();

function qs(params = {}) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    p.set(k, Array.isArray(v) ? v.join("|") : String(v));
  }
  const s = p.toString();
  return s ? `?${s}` : "";
}

export async function get(path, params, { fresh = false } = {}) {
  const url = `api/${path}${qs(params)}`;
  if (!fresh && cache.has(url)) return cache.get(url);
  const req = fetch(url).then(async (r) => {
    const body = await r.json().catch(() => null);
    if (!r.ok) throw new ApiError(r.status, body);
    return body;
  });
  cache.set(url, req);
  req.catch(() => cache.delete(url));
  return req;
}

export async function post(path, body) {
  const r = await fetch(`api/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => null);
  if (!r.ok) throw new ApiError(r.status, data);
  return data;
}
