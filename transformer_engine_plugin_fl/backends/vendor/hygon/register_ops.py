# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Hygon vendor backend operator registrations."""

from __future__ import annotations

from ..op_names import ALL_OPS, OPTIONAL_OPS
from ..registration import register_vendor_ops


def register_builtins(registry) -> None:
    """Register all Hygon (VENDOR) operator implementations."""
    from .hygon import HygonBackend

    register_vendor_ops(
        registry,
        HygonBackend(),
        ALL_OPS,
        impl_id="vendor.hygon",
        vendor="HYGON",
        optional_ops=OPTIONAL_OPS,
    )
