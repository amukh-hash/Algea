export function auditLog(action: string, detail?: Record<string, unknown>) {
  if (process.env.NODE_ENV !== "production") {
    console.info("[audit]", action, detail ?? {});
  }
}
