"""
zmq_auth.py — CurveZMQ Key Lifecycle Management

Generates and persists a Curve25519 keypair for the ZMQ XPUB sockets.
The public key is served via /api/control/state for REST-bootstrapped
key exchange over the existing TLS 1.3 tunnel.

Keys persist across standard backend restarts to prevent stranding
active C++ clients. Manual key deletion forces regeneration on restart,
which C++ clients detect via REST polling mismatch and reconnect.

Usage:
    from backend.app.api.zmq_auth import initialize_curve_keys, get_server_public_key
    server_public, server_secret = initialize_curve_keys()
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import zmq.auth

logger = logging.getLogger("algae.zmq.auth")

# Default key directory — override via ALGAE_CURVE_DIR env var
CURVE_DIR = os.environ.get("ALGAE_CURVE_DIR", os.path.join(os.path.expanduser("~"), ".algae", "crypto"))
SERVER_KEY_NAME = "server"


def initialize_curve_keys(curve_dir: str | None = None) -> tuple[bytes, bytes]:
    """Generate or load a Curve25519 Z85-encoded keypair.

    Returns:
        (server_public_key, server_secret_key) as Z85-encoded bytes.
    """
    key_dir = Path(curve_dir or CURVE_DIR)
    public_file = key_dir / f"{SERVER_KEY_NAME}.key"
    secret_file = key_dir / f"{SERVER_KEY_NAME}.key_secret"

    if not (public_file.exists() and secret_file.exists()):
        logger.info("CurveZMQ: generating new Curve25519 keypair in %s", key_dir)
        key_dir.mkdir(parents=True, exist_ok=True)
        zmq.auth.create_certificates(str(key_dir), SERVER_KEY_NAME)
    else:
        logger.info("CurveZMQ: loaded existing keypair from %s", key_dir)

    server_public, server_secret = zmq.auth.load_certificate(str(secret_file))
    logger.info("CurveZMQ server public key: %s", server_public.decode("utf-8")[:8] + "...")
    return server_public, server_secret


def get_server_public_key(curve_dir: str | None = None) -> str:
    """Return the server public key as a UTF-8 string for REST distribution."""
    key_dir = Path(curve_dir or CURVE_DIR)
    secret_file = key_dir / f"{SERVER_KEY_NAME}.key_secret"

    if not secret_file.exists():
        raise FileNotFoundError(f"CurveZMQ keypair not found at {secret_file}")

    server_public, _ = zmq.auth.load_certificate(str(secret_file))
    return server_public.decode("utf-8")


def apply_curve_server(socket, curve_dir: str | None = None) -> None:
    """Apply CurveZMQ server-side configuration to a ZMQ socket.

    Call this BEFORE socket.bind().
    """
    import zmq as _zmq

    _, server_secret = initialize_curve_keys(curve_dir)
    socket.setsockopt(_zmq.CURVE_SERVER, True)
    socket.setsockopt(_zmq.CURVE_SECRETKEY, server_secret)
    logger.info("CurveZMQ: server-side encryption enabled on socket")
