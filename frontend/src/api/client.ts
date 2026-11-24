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

/** PUT a file as the request body to ``path`` (under /api); the server's reason comes back as the error. */
async function putFile<T>(path: string, file: File, contentType: string): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method: "PUT",
    headers: { "X-Requested-With": "rtool", "Content-Type": contentType },
    body: file,
    credentials: "same-origin",
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      // keep the status text
    }
    throw new ApiError(response.status, String(detail));
  }
  return (await response.json()) as T;
}

const runApi = (project: string, runId: string) => `/runs/${encodeURIComponent(project)}/${encodeURIComponent(runId)}`;

/** Send a paper the researcher has (PDF or HTML) into the literature stage's inbox. */
export function uploadPaper<T>(
  project: string,
  runId: string,
  file: File,
  fields: { title: string; paper_id?: string; doi?: string },
): Promise<T> {
  const params = new URLSearchParams(
    Object.entries(fields).filter((entry): entry is [string, string] => Boolean(entry[1])),
  );
  const html = /\.html?$/i.test(file.name) || file.type === "text/html";
  return putFile(`${runApi(project, runId)}/literature/inbox?${params}`, file, html ? "text/html" : "application/pdf");
}

/** Send the study's data.csv into the collect stage's inbox (the server checks it against the protocol). */
export function uploadStudyData<T>(project: string, runId: string, file: File): Promise<T> {
  return putFile(`${runApi(project, runId)}/collect/inbox`, file, "text/csv");
}

/** In-app reader for one paper (optionally scrolled to one of its claims). */
/** The in-app reader for a paper, optionally at one of its claims or at a quote (e.g. a hypothesis's evidence). */
export function readerPath(
  project: string,
  runId: string,
  stage: string,
  paperId: string,
  at?: number | { quote: string },
): string {
  const query = new URLSearchParams({ paper: paperId });
  if (typeof at === "number") query.set("claim", String(at));
  else if (at) query.set("quote", at.quote);
  return `${runPath(project, runId)}/${encodeURIComponent(stage)}/reader?${query}`;
}
