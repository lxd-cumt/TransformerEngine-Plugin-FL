# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Guard the op API contract shared by every backend.

``TEFLBackendBase`` tracks the pybind surface of TransformerEngine main. A
backend that overrides an op with a stale parameter list still imports fine and
only fails when that op is dispatched, so the mismatch is checked statically
here instead.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from transformer_engine_plugin_fl.ops import TEFLBackendBase
from transformer_engine_plugin_fl.backends.vendor.op_names import ALL_OPS, OPTIONAL_OPS

# Backends expected to implement the full op surface: they forward every op to a
# vendor extension built from the same source as upstream.
FULL_SURFACE_BACKENDS = [
    ("vendor.cuda.cuda", "CUDABackend"),
    ("vendor.enflame.enflame", "EnflameBackend"),
    ("vendor.hygon.hygon", "HygonBackend"),
    ("vendor.iluvatar.iluvatar", "IluvatarBackend"),
    ("vendor.metax.metax", "MetaxBackend"),
    ("vendor.musa.musa", "MUSABackend"),
]

# Backends implementing a deliberate subset, falling through the manager for the
# rest. They must still match the base signature for whatever they do override.
PARTIAL_BACKENDS = [
    ("flagos.flagos", "FlagOSBackend"),
    ("reference.reference", "ReferenceBackend"),
    ("vendor.kunlunxin.kunlunxin", "KunLunXinBackend"),
    ("vendor.npu.npu", "NPUBackend"),
]

ALL_BACKENDS = FULL_SURFACE_BACKENDS + PARTIAL_BACKENDS


def _load(path: str, cls_name: str):
    module = importlib.import_module(f"transformer_engine_plugin_fl.backends.{path}")
    return getattr(module, cls_name)


def _signature(cls, name):
    """Parameter names, order and kind -- the part dispatch depends on.

    Type annotations are deliberately excluded: a backend narrowing
    ``Optional[Any]`` to ``Optional[torch.device]`` documents its own extension
    and cannot break a call, whereas a renamed or reordered parameter does.
    """
    params = inspect.signature(getattr(cls, name)).parameters
    return tuple((p.name, p.kind, p.default is not inspect.Parameter.empty) for p in params.values())


def _base_ops():
    return {
        name
        for name, fn in inspect.getmembers(TEFLBackendBase, inspect.isfunction)
        if not name.startswith("_")
    }


@pytest.mark.parametrize("path,cls_name", ALL_BACKENDS, ids=[c for _, c in ALL_BACKENDS])
def test_signatures_match_base(path, cls_name):
    """Every overridden op keeps the base's parameter names and order."""
    cls = _load(path, cls_name)
    mismatches = []
    for op in sorted(_base_ops()):
        own = cls.__dict__.get(op)
        if not inspect.isfunction(own):
            continue  # inherited, so trivially in sync
        base_sig = _signature(TEFLBackendBase, op)
        if _signature(cls, op) != base_sig:
            mismatches.append(f"{op}:\n  base: {base_sig}\n  impl: {_signature(cls, op)}")
    assert not mismatches, f"{cls_name} drifted from TEFLBackendBase:\n" + "\n".join(mismatches)


@pytest.mark.parametrize(
    "path,cls_name", FULL_SURFACE_BACKENDS, ids=[c for _, c in FULL_SURFACE_BACKENDS]
)
def test_full_surface_backends_implement_every_op(path, cls_name):
    """Passthrough backends expose the whole op surface, not a subset."""
    cls = _load(path, cls_name)
    missing = sorted(op for op in _base_ops() if not hasattr(cls, op))
    assert not missing, f"{cls_name} is missing {len(missing)} ops: {missing}"


def test_base_covers_the_registered_op_list():
    """``ALL_OPS`` and the base class describe the same surface."""
    base = _base_ops()
    # `is_available` is backend plumbing rather than a dispatchable op.
    assert sorted(set(ALL_OPS) - base) == []
    assert sorted(base - set(ALL_OPS)) == ["is_available"]


def test_optional_ops_are_part_of_the_surface():
    """Conditionally-compiled ops are still declared, just registered lazily."""
    assert set(OPTIONAL_OPS) <= set(ALL_OPS)
    assert all(op.startswith("ep_") for op in OPTIONAL_OPS)


@pytest.mark.parametrize("path,cls_name", ALL_BACKENDS, ids=[c for _, c in ALL_BACKENDS])
def test_backend_is_constructible_without_its_hardware(path, cls_name):
    """Constructing a backend must not require its extension to be present.

    Registration instantiates every backend to ask whether it is available, so a
    constructor that imports the vendor extension eagerly would break startup on
    all other hardware.
    """
    cls = _load(path, cls_name)
    assert not cls.__abstractmethods__, f"{cls_name} leaves {cls.__abstractmethods__} abstract"
    cls()
