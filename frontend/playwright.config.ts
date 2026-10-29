/** End-to-end tests: the built app against a real `rtool ui --mode local` server (demo fakes, no model, no cluster).
 *
 *     npm run build && npm run e2e
 *
 * Playwright's own Chromium does not run on macOS 13, so the installed Google Chrome is used (with a fresh,
 * temporary profile).
 */
import { defineConfig } from "@playwright/test";

const PORT = 8799;

export default defineConfig({
  testDir: "e2e",
  timeout: 300_000,
  expect: { timeout: 60_000 },
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: `http://127.0.0.1:${PORT}`, channel: "chrome", headless: true, trace: "retain-on-failure" },
  webServer: {
    command: `rm -rf .e2e && mkdir -p .e2e/runs && ../.venv/bin/rtool ui --mode local --runs-root .e2e/runs --port ${PORT} --token-file .e2e/token`,
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
