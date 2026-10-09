// Records a hash of the frontend sources next to the built app, so a Python test can tell
// when the committed build is older than the sources (tests/web/test_frontend_bundle.py).
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const SOURCES = ["index.html", "package.json", "package-lock.json", "vite.config.ts", "tsconfig.json", "src", "public"];

function files(path) {
  const stats = statSync(path, { throwIfNoEntry: false });
  if (!stats) return [];
  if (stats.isFile()) return [path];
  return readdirSync(path).flatMap((name) => files(join(path, name)));
}

const paths = SOURCES.flatMap((p) => files(join(root, p)))
  .map((p) => relative(root, p).split(sep).join("/"))
  .filter((p) => !p.includes(".test.") && !p.startsWith("src/test/"))
  .sort();
const hash = createHash("sha256");
for (const path of paths) {
  hash.update(path + "\0");
  hash.update(readFileSync(join(root, path)));
  hash.update("\0");
}
const stamp = { sources_sha256: hash.digest("hex"), files: paths.length };
writeFileSync(join(root, "../src/research_tool/web/static/build-stamp.json"), JSON.stringify(stamp, null, 2) + "\n");
console.log(`build stamp: ${stamp.sources_sha256.slice(0, 12)} (${stamp.files} files)`);
