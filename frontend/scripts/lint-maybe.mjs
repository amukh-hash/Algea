#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

function has(moduleName) {
  try {
    require.resolve(`${moduleName}/package.json`);
    return true;
  } catch {
    return false;
  }
}

const hasEslint = has("eslint");
const hasNextConfig = has("eslint-config-next");

if (!hasEslint || !hasNextConfig) {
  console.log(
    "[lint:maybe] Skipping lint: missing local devDependencies (eslint and/or eslint-config-next). " +
      "This is expected in restricted environments without npm registry access."
  );
  process.exit(0);
}

const result = spawnSync("npm", ["run", "lint:ci"], {
  stdio: "inherit",
  shell: process.platform === "win32",
});

if (typeof result.status === "number") {
  process.exit(result.status);
}

process.exit(1);
