import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        app: "var(--color-bg)",
        "surface-1": "var(--color-surface-1)",
        "surface-2": "var(--color-surface-2)",
        primary: "var(--color-text)",
        secondary: "var(--color-text-secondary)",
        muted: "var(--color-text-muted)",
        border: "var(--color-border)",
        "border-subtle": "var(--color-border-subtle)",
        success: "var(--color-success)",
        warning: "var(--color-warning)",
        error: "var(--color-error)",
        info: "var(--color-info)"
      },
      borderRadius: { md: "var(--radius-md)" }
    },
  },
  plugins: [],
} satisfies Config;
