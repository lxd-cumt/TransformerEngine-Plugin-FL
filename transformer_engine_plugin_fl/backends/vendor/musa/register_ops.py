# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""MUSA vendor backend operator registrations."""

from __future__ import annotations

from ..op_names import ALL_OPS, OPTIONAL_OPS
from ..registration import register_vendor_ops


def register_builtins(registry) -> None:
    """Register all MUSA (VENDOR) operator implementations."""
    from .musa import MUSABackend

    register_vendor_ops(
        registry,
        MUSABackend(),
        ALL_OPS,
        impl_id="vendor.musa",
        vendor="MUSA",
        optional_ops=OPTIONAL_OPS,
    )
