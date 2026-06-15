"""Unified patch interface — delegates to backend-specific patches.

TransformerEngine calls `plugin.patches.apply_patches()` without
needing to know which vendor backend is active.
"""

from __future__ import annotations


def apply_patches() -> None:
    """Discover and apply patches from the active backend (best-effort)."""
    _try_musa_patches()
    # Future backends: add _try_xxx_patches() calls here.


def _try_musa_patches() -> None:
    try:
        from transformer_engine_plugin_fl.backends.musa.patches import apply_patch

        apply_patch()
    except Exception:
        # Not on MUSA or backend unavailable — silently skip.
        pass
