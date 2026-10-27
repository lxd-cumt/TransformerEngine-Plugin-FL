# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Built-in operator implementations registration.

This module registers DEFAULT (FlagOS) and REFERENCE (PyTorch) implementations
for all supported operators by calling register_builtins from each backend.
"""

from __future__ import annotations

import importlib

from .registry import OpRegistry

# Vendor backends, registered in order. A backend whose hardware or extension
# is absent raises on import or reports itself unavailable; that is the normal
# case on any given machine, so the failure is not surfaced.
VENDOR_BACKENDS = (
    "cuda",
    "enflame",
    "hygon",
    "iluvatar",
    "kunlunxin",
    "metax",
    "musa",
    "npu",
)


def register_builtins(registry: OpRegistry) -> None:
    """
    Register all built-in operator implementations.

    This function registers:
    - DEFAULT implementations (FlagOS/flag_gems)
    - REFERENCE implementations (PyTorch)
    - VENDOR implementations (one per supported accelerator, when available)

    Args:
        registry: Registry to register into
    """
    # DEFAULT (FlagOS) and REFERENCE (PyTorch) are always expected to load, so
    # a failure here is worth reporting.
    for name, label in (("flagos", "FlagOS"), ("reference", "Reference")):
        try:
            module = importlib.import_module(f".backends.{name}.register_ops", __package__)
            module.register_builtins(registry)
        except Exception as e:  # pragma: no cover - depends on optional deps
            print(f"[WARNING] Failed to register {label} operators: {e}")

    for vendor in VENDOR_BACKENDS:
        try:
            module = importlib.import_module(
                f".backends.vendor.{vendor}.register_ops", __package__
            )
            module.register_builtins(registry)
        except Exception:
            # The vendor's hardware or extension is not present here.
            pass
