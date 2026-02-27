"""Compatibility package for the Algae 4.0 rebrand.

This module re-exports the legacy ``algaie`` package namespace so existing imports
continue to work while allowing new tooling to reference ``algae``.
"""

from algaie import *  # noqa: F401,F403
