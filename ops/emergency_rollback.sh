#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# emergency_rollback.sh — Atomic Reverse-Proxy Rollback
#
# Restores the legacy Next.js frontend within milliseconds by swapping
# the Nginx active config symlink. Does NOT edit config files directly.
#
# Usage:  sudo ./emergency_rollback.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SITES_AVAILABLE="/etc/nginx/sites-available"
SITES_ENABLED="/etc/nginx/sites-enabled"
ACTIVE_LINK="${SITES_ENABLED}/algae_active.conf"
LEGACY_CONF="${SITES_AVAILABLE}/algae_legacy.conf"
SUNSET_CONF="${SITES_AVAILABLE}/algae_sunset.conf"
BACKEND_URL="http://127.0.0.1:8000"

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  EMERGENCY ROLLBACK — Restoring Legacy Next.js UI    ║"
echo "╚═══════════════════════════════════════════════════════╝"

# Verify legacy config exists
if [ ! -f "$LEGACY_CONF" ]; then
    echo "FATAL: Legacy config not found at $LEGACY_CONF"
    exit 1
fi

# 1. Atomic symlink swap (single rename syscall — no intermediate state)
echo "[1/4] Swapping symlink to legacy config..."
ln -sfn "$LEGACY_CONF" "$ACTIVE_LINK"

# 2. Syntax validation before reload
echo "[2/4] Validating Nginx configuration..."
if ! nginx -t 2>&1; then
    echo "FATAL: Nginx syntax invalid. Reverting symlink to sunset config."
    ln -sfn "$SUNSET_CONF" "$ACTIVE_LINK"
    exit 1
fi

# 3. Graceful reload — does not drop active TCP keep-alives
echo "[3/4] Reloading Nginx workers..."
systemctl reload nginx

# 4. Force legacy clients to drop stale React TanStack Query caches
echo "[4/4] Broadcasting state invalidation to backend..."
curl -s -X POST "${BACKEND_URL}/api/control/force-sync" \
    -H "Content-Type: application/json" \
    -d '{"reason": "emergency_rollback"}' \
    || echo "WARNING: Backend force-sync failed (may be unreachable)"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ROLLBACK COMPLETE"
echo "  Legacy Next.js UI is now active."
echo "  Monitor: https://trade.algae.internal/"
echo "═══════════════════════════════════════════════════════"
