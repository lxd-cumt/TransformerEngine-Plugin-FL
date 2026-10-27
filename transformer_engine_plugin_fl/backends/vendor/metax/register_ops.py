# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Metax vendor backend operator registrations."""

from __future__ import annotations

from ..op_names import ALL_OPS, OPTIONAL_OPS
from ..registration import register_vendor_ops


def register_builtins(registry) -> None:
    """Register all Metax (VENDOR) operator implementations."""
    from .metax import MetaxBackend

    register_vendor_ops(
        registry,
        MetaxBackend(),
        ALL_OPS,
        impl_id="vendor.metax",
        vendor="METAX",
        optional_ops=OPTIONAL_OPS,
    )
