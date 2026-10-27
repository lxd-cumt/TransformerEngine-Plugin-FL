# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Iluvatar vendor backend operator registrations."""

from __future__ import annotations

from ..op_names import ALL_OPS, OPTIONAL_OPS
from ..registration import register_vendor_ops


def register_builtins(registry) -> None:
    """Register all Iluvatar (VENDOR) operator implementations."""
    from .iluvatar import IluvatarBackend

    register_vendor_ops(
        registry,
        IluvatarBackend(),
        ALL_OPS,
        impl_id="vendor.iluvatar",
        vendor="Iluvatar",
        optional_ops=OPTIONAL_OPS,
    )
