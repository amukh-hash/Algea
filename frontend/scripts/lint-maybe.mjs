#!/usr/bin/env node
// ---------------------------------------------------------------------------
// lint-maybe.mjs — Conditional lint runner for restricted environments
//
// Usage:
//   npm run lint:ci     — Use in CI. Runs ESLint via Next.js with
//                         --max-warnings=0. Fails on any warning or error.
//   npm run lint:maybe  — Use in sandboxed / restricted environments where
//                         npm registry access may be blocked. Skips lint
//                         gracefully (exit 0) when eslint or
//                         eslint-config-next are not installed; otherwise
//                         delegates to lint:ci and propagates its exit code.
// ---------------------------------------------------------------------------

import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

/**
 * Check whether a package is resolvable from the current working tree.
 * Uses the package.json sub-path to avoid triggering module side-effects.
 * @param {string} moduleName  npm package name (e.g. "eslint")
 * @returns {boolean}
 */
function has(moduleName) {
  try {
    require.resolve(`${moduleName}/package.json`);
    return true;
  } catch {
    return false;
  }
}

// ---- Dependency detection ---------------------------------------------------

const hasEslint = has("eslint");
const hasNextConfig = has("eslint-config-next");

if (!hasEslint || !hasNextConfig) {
  const missing = [];
  if (!hasEslint) missing.push("eslint");
  if (!hasNextConfig) missing.push("eslint-config-next");

  console.log(
    `[lint:maybe] Skipping lint — missing devDependencies: ${missing.join(", ")}.\n` +
    "  This is expected in restricted environments without npm registry access.\n" +
    "  To enable linting, run `npm install` with registry access and use `npm run lint:ci`."
  );
  process.exit(0);
}

// ---- Delegate to lint:ci ----------------------------------------------------

const result = spawnSync("npm", ["run", "lint:ci"], {
  stdio: "inherit",
  shell: process.platform === "win32",
});

if (typeof result.status === "number") {
  process.exit(result.status);
}

// spawn failed without producing an exit code (e.g. ENOENT)
const reason = result.error ? result.error.message : "unknown error";
console.error(`[lint:maybe] Failed to run lint:ci — ${reason}`);
process.exit(1);
