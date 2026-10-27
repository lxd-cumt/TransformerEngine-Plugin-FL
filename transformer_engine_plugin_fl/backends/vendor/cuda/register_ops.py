# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""CUDA vendor backend operator registrations."""

from __future__ import annotations

from ..op_names import ALL_OPS, OPTIONAL_OPS
from ..registration import register_vendor_ops


def register_builtins(registry) -> None:
    """Register all CUDA (VENDOR) operator implementations."""
    from .cuda import CUDABackend

    register_vendor_ops(
        registry,
        CUDABackend(),
        ALL_OPS,
        impl_id="vendor.cuda",
        vendor="CUDA",
        optional_ops=OPTIONAL_OPS,
    )
