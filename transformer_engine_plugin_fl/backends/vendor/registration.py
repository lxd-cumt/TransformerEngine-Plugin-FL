# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Shared op-registration helper for the vendor backends.

Every vendor op is a uniform passthrough to the matching backend method, so the
registrations are generated from a name list rather than spelled out one
``OpImpl`` at a time: all entries for a vendor share the same ``impl_id``,
``vendor`` and ``priority``, so per-op boilerplate carries no information.
"""

from __future__ import annotations

import functools
from typing import Iterable, Optional

from ...types import OpImpl, BackendImplKind


def bind_is_available(fn, is_available_fn):
    """Wrap ``fn`` so ``OpImpl.is_available()`` can consult the backend."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_vendor_ops(
    registry,
    backend,
    op_names: Iterable[str],
    *,
    impl_id: str,
    vendor: str,
    priority: int = 100,
    optional_ops: Iterable[str] = (),
    kind: BackendImplKind = BackendImplKind.VENDOR,
) -> int:
    """Register ``op_names`` as passthroughs to ``backend``'s methods.

    Ops the backend does not implement are skipped. Ops listed in
    ``optional_ops`` are additionally skipped when the loaded extension lacks
    the symbol -- they are conditionally compiled upstream (for example the
    ``ep_*`` ops behind ``NVTE_WITH_NCCL_EP``), so their absence is expected
    rather than an error.

    Returns the number of ops registered.
    """
    if not backend.is_available():
        return 0

    is_avail = backend.is_available
    optional = set(optional_ops)
    tex: Optional[object] = None
    if optional:
        try:
            tex = backend._get_tex()
        except Exception:
            tex = None

    impls = []
    for op_name in op_names:
        method = getattr(backend, op_name, None)
        if method is None:
            continue
        if op_name in optional and not hasattr(tex, op_name):
            continue
        impls.append(
            OpImpl(
                op_name=op_name,
                impl_id=impl_id,
                kind=kind,
                fn=bind_is_available(method, is_avail),
                vendor=vendor,
                priority=priority,
            )
        )

    registry.register_many(impls)
    return len(impls)
