import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        app: "var(--colors-bg)",
        "surface-1": "var(--colors-surface1)",
        "surface-2": "var(--colors-surface2)",
        primary: "var(--colors-text)",
        secondary: "var(--colors-textSecondary)",
        muted: "var(--colors-textMuted)",
        border: "var(--colors-border)",
        "border-subtle": "var(--colors-borderSubtle)",
        success: "var(--colors-success)",
        warning: "var(--colors-warning)",
        danger: "var(--colors-error)",
        info: "var(--colors-info)",
        focus: "var(--colors-focus)",
      },
      borderRadius: {
        sm: "var(--radius-sm)",
        md: "var(--radius-md)",
        lg: "var(--radius-lg)",
        xl: "var(--radius-xl)",
      },
      fontFamily: {
        sans: "var(--typography-font-sans)",
      },
    },
  },
  plugins: [],
} satisfies Config;
