"""Unified patch interface — delegates to backend-specific patches.

TransformerEngine calls `plugin.patches.apply_patches()` without
needing to know which vendor backend is active.
"""

from __future__ import annotations

import importlib

# Vendors shipping Python-side patches. Each module exposes `apply_patch()`,
# which must decide for itself whether its hardware is present -- only one of
# these is ever active on a given machine.
PATCHED_VENDORS = ("musa", "npu", "tsingmicro")


def apply_patches() -> None:
    """Discover and apply patches from the active backend (best-effort)."""
    for vendor in PATCHED_VENDORS:
        try:
            module = importlib.import_module(
                f"transformer_engine_plugin_fl.backends.vendor.{vendor}.patches"
            )
            module.apply_patch()
        except Exception:
            # Not on this vendor, or the backend is unavailable — skip.
            continue
