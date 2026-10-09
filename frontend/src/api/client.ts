/** Talking to `rtool ui`: JSON over fetch, with the session cookie and the header writes need. */

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
  }
}

const signedOutListeners = new Set<() => void>();

/** Called when the server says we are not signed in (expired session, restarted server). */
export function onSignedOut(listener: () => void): () => void {
  signedOutListeners.add(listener);
  return () => signedOutListeners.delete(listener);
}

function describe(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc: [...], msg: "..."}]
    return detail
      .map((item) => {
        const where = Array.isArray(item?.loc) ? item.loc.filter((p: unknown) => p !== "body").join(".") : "";
        return where ? `${where}: ${item?.msg}` : String(item?.msg ?? item);
      })
      .join("; ");
  }
  return JSON.stringify(detail);
}

export async function api<T>(
  path: string,
  options: { method?: "GET" | "POST" | "PUT" | "DELETE"; body?: unknown; signal?: AbortSignal } = {},
): Promise<T> {
  const method = options.method ?? "GET";
  const headers: Record<string, string> = { Accept: "application/json" };
  if (method !== "GET") headers["X-Requested-With"] = "rtool";
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(`/api${path}`, {
    method,
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    credentials: "same-origin",
    signal: options.signal,
  });
  if (response.status === 401 && path !== "/session") signedOutListeners.forEach((listener) => listener());
  if (!response.ok) {
    let detail = response.statusText || `HTTP ${response.status}`;
    try {
      detail = describe((await response.json()).detail);
    } catch {
      // not JSON: keep the status text
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204 || response.status === 304) return undefined as T;
  return (await response.json()) as T;
}

export function signIn(token: string): Promise<void> {
  return api<void>("/session", { method: "POST", body: { token } });
}

/** URL of a file inside a run directory (papers, figures, logs), for links and frames. */
export function fileUrl(project: string, runId: string, path: string): string {
  return `/api/runs/${encodeURIComponent(project)}/${encodeURIComponent(runId)}/files/${path
    .split("/")
    .map(encodeURIComponent)
    .join("/")}`;
}

export function runPath(project: string, runId: string): string {
  return `/runs/${encodeURIComponent(project)}/${encodeURIComponent(runId)}`;
}
