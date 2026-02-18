import { readFileSync, writeFileSync } from "fs";
import { join } from "path";

const root = join(process.cwd(), "src/theme/tokens");
const base = JSON.parse(readFileSync(join(root, "tokens.base.json"), "utf8"));
const dark = JSON.parse(readFileSync(join(root, "tokens.dark.json"), "utf8"));
const light = JSON.parse(readFileSync(join(root, "tokens.light.json"), "utf8"));

const css = `:root {\n  --font-sans: ${base.typography.fontSans};\n  --space-4: ${base.spacing["4"]};\n  --radius-md: ${base.radius.md};\n}\n[data-theme="dark"] {\n  --color-bg: ${dark.color.bg};\n  --color-surface-1: ${dark.color.surface1};\n  --color-surface-2: ${dark.color.surface2};\n  --color-border: ${dark.color.border};\n  --color-border-subtle: ${dark.color.borderSubtle};\n  --color-text: ${dark.color.text};\n  --color-text-secondary: ${dark.color.textSecondary};\n  --color-text-muted: ${dark.color.textMuted};\n  --color-focus: ${dark.color.focus};\n  --color-success: ${dark.color.success};\n  --color-warning: ${dark.color.warning};\n  --color-error: ${dark.color.error};\n  --color-info: ${dark.color.info};\n  --series-1: ${dark.series[0]};\n  --series-2: ${dark.series[1]};\n  --series-3: ${dark.series[2]};\n  --series-4: ${dark.series[3]};\n  --series-5: ${dark.series[4]};\n  --series-6: ${dark.series[5]};\n}\n[data-theme="light"] {\n  --color-bg: ${light.color.bg};\n}\n`;
writeFileSync(join(process.cwd(), "src/theme/generated.css"), css);
