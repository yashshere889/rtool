/** Records the demo video against a running `rtool ui --mode local --demo` (see README.md). Not part of `npm run e2e`. */
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  testMatch: "*.rec.ts",
  timeout: 1_800_000,
  expect: { timeout: 120_000 },
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${process.env.DEMO_PORT ?? 8811}`,
    channel: "chrome",
    headless: true,
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
    actionTimeout: 60_000,
  },
});
