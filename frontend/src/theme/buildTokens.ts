import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

type TokenMap = Record<string, any>;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const base = JSON.parse(readFileSync(path.join(__dirname, "tokens/tokens.base.json"), "utf8")) as TokenMap;
const dark = JSON.parse(readFileSync(path.join(__dirname, "tokens/tokens.dark.json"), "utf8")) as TokenMap;
const light = JSON.parse(readFileSync(path.join(__dirname, "tokens/tokens.light.json"), "utf8")) as TokenMap;

function flatten(obj: TokenMap, prefix = ""): Record<string, string> {
  return Object.entries(obj).reduce<Record<string, string>>((acc, [key, value]) => {
    const next = prefix ? `${prefix}-${key}` : key;
    if (typeof value === "string" || typeof value === "number") acc[next] = String(value);
    else Object.assign(acc, flatten(value, next));
    return acc;
  }, {});
}

const toCss = (selector: string, data: TokenMap) => {
  const flat = flatten(data);
  const vars = Object.entries(flat)
    .map(([k, v]) => `  --${k.replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`)}: ${v};`)
    .join("\n");
  return `${selector} {\n${vars}\n}`;
};

const output = [toCss(":root", { ...base, ...light }), toCss('[data-theme="dark"]', { ...base, ...dark })].join("\n\n");
writeFileSync(path.join(__dirname, "generated.css"), output);
