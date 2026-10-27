# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Enflame vendor backend operator registrations."""

from __future__ import annotations

from ..op_names import ALL_OPS, OPTIONAL_OPS
from ..registration import register_vendor_ops


def register_builtins(registry) -> None:
    """Register all Enflame (VENDOR) operator implementations."""
    from .enflame import EnflameBackend

    register_vendor_ops(
        registry,
        EnflameBackend(),
        ALL_OPS,
        impl_id="vendor.enflame",
        vendor="ENFLAME",
        optional_ops=OPTIONAL_OPS,
    )
