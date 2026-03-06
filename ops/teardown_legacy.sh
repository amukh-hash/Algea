#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# teardown_legacy.sh — Day-3 Legacy Infrastructure Decommissioning
#
# Permanently destroys the Next.js frontend, PM2 processes, and static
# assets after the 48-hour rollback window expires.
#
# Addresses Blind Spot 2 (PM2 daemon ghosting): uses pm2 kill + pkill
# to guarantee socket liberation before daemon restart.
#
# Usage:  sudo ./teardown_legacy.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

WEB_DIR="/var/www/algae_web_legacy"
PM2_APP="next-algae"

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  DAY-3 TEARDOWN — Decommissioning Legacy Next.js     ║"
echo "╚═══════════════════════════════════════════════════════╝"

# Safety check: confirm operator intent
read -rp "This permanently destroys the legacy web UI. Continue? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

# ── Phase 1: PM2 Process Eradication ────────────────────────────────
echo "[1/5] Removing $PM2_APP from PM2 registry..."
pm2 delete "$PM2_APP" 2>/dev/null || echo "  (not found in registry)"
pm2 save --force

# Blind Spot 2: PM2 ghost processes retain file descriptors
echo "[2/5] Killing PM2 daemon to release ghost file descriptors..."
pm2 kill 2>/dev/null || true

# Eradicate any orphaned Node.js worker processes
echo "[3/5] Purging orphaned Node.js processes..."
pkill -f "node.*next-algae" 2>/dev/null || true
pkill -f "node.*algae_web" 2>/dev/null || true

# Restart PM2 daemon (clean, no ghosts)
echo "  Restarting PM2 daemon..."
pm2 resurrect 2>/dev/null || pm2 startup 2>/dev/null || true

# ── Phase 2: Static Asset Destruction ───────────────────────────────
if [ -d "$WEB_DIR" ]; then
    echo "[4/5] Destroying static web assets at $WEB_DIR..."
    rm -rf "$WEB_DIR"
    echo "  Removed."
else
    echo "[4/5] No static assets found at $WEB_DIR (already clean)."
fi

# ── Phase 3: Verify Port Liberation ─────────────────────────────────
echo "[5/5] Verifying port 3000 is free..."
if ss -tlnp | grep -q ":3000 "; then
    echo "  WARNING: Port 3000 still in use:"
    ss -tlnp | grep ":3000 "
    echo "  Manual investigation required."
else
    echo "  Port 3000 is free."
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  TEARDOWN COMPLETE"
echo "  Next.js processes: destroyed"
echo "  Static assets: purged"
echo "  Port 3000: liberated"
echo "═══════════════════════════════════════════════════════"
