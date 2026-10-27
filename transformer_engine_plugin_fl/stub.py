# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Stub module for transformer_engine_torch when CUDA pybind is unavailable.

When NVTE_SKIP_CUDA=1, the original pybind .so cannot be compiled/loaded.
This module registers itself as sys.modules["transformer_engine_torch"],
providing:
  - All enums (DType, NVTE_Bias_Type, etc.) as pure Python IntEnums
  - All op functions dispatched through OpManager
  - Class stubs (FP8TensorMeta, CommOverlap, etc.)

TE upper-layer code can reference tex.DType.kFloat32 etc. without change.
"""

from __future__ import annotations

import sys
import types
import logging

from .ops import (
    DType,
    Float8BlockScaleTensorFormat,
    NVTE_Activation_Type,
    NVTE_Softmax_Type,
    CommGemmOverlapRole,
    FP8FwdTensors,
    FP8BwdTensors,
    NVTE_Bias_Type,
    NVTE_Mask_Type,
    NVTE_Fused_Attn_Backend,
    NVTE_QKV_Format,
    NVTE_QKV_Layout,
    NVTERoutingMapFormat,
    NVTEQBHistogramMode,
    NVTEScalingMode,
    CommOverlapType,
    CommOverlapAlgo,
    FP8TensorMeta,
    CommOverlapHelper,
    CommOverlap,
    CommOverlapP2P,
    CommOverlapBase,
    CommOverlapCore,
    CommOverlapP2PBase,
    TEFLBackendBase,
)

logger = logging.getLogger(__name__)


def _te_main_op_names() -> list[str]:
    """Op names the stub must expose, taken from the TEFLBackendBase interface.

    ``TEFLBackendBase`` mirrors TE main's pybind surface, so reusing it avoids a
    second list that could fall out of sync. The plugin-internal hooks (backend
    selection, class factories) are not pybind ops and are filtered out.
    """
    skip = {
        "is_available",
        "get_attention_backend",
        "get_flash_attention_class",
        "create_fp8_tensor_meta",
        "create_comm_overlap",
        "create_comm_overlap_helper",
        "create_comm_overlap_p2p",
        "convert_host_pointers_to_tensor",
        "get_device_pointer_for_data_and_scales",
    }
    return [
        name
        for name in vars(TEFLBackendBase)
        if not name.startswith("_") and name not in skip and callable(getattr(TEFLBackendBase, name))
    ]


def _make_op_dispatcher(op_name: str):
    """Create a dispatcher function that routes to OpManager."""

    def dispatcher(*args, **kwargs):
        from .manager import get_default_manager

        return get_default_manager().call(op_name, *args, **kwargs)

    dispatcher.__name__ = op_name
    dispatcher.__qualname__ = f"transformer_engine_torch.{op_name}"
    return dispatcher


def install_stub(manager) -> None:
    """Install a stub module as sys.modules["transformer_engine_torch"].

    Always takes over transformer_engine_torch. The stub exposes all enums
    and dispatches op calls through OpManager. CUDA backend separately imports
    transformer_engine_torch_nv (the original pybind .so) to register its ops.
    """
    module_name = "transformer_engine_torch"

    stub = types.ModuleType(module_name)
    stub.__doc__ = "TE Plugin stub module (enums + OpManager dispatch)"
    stub.__package__ = module_name

    # Register all enums
    stub.DType = DType
    stub.Float8BlockScaleTensorFormat = Float8BlockScaleTensorFormat
    stub.NVTE_Activation_Type = NVTE_Activation_Type
    stub.NVTE_Softmax_Type = NVTE_Softmax_Type
    stub.CommGemmOverlapRole = CommGemmOverlapRole
    stub.FP8FwdTensors = FP8FwdTensors
    stub.FP8BwdTensors = FP8BwdTensors
    stub.NVTE_Bias_Type = NVTE_Bias_Type
    stub.NVTE_Mask_Type = NVTE_Mask_Type
    stub.NVTE_Fused_Attn_Backend = NVTE_Fused_Attn_Backend
    stub.NVTE_QKV_Format = NVTE_QKV_Format
    stub.NVTE_QKV_Layout = NVTE_QKV_Layout
    stub.NVTERoutingMapFormat = NVTERoutingMapFormat
    stub.NVTEQBHistogramMode = NVTEQBHistogramMode
    stub.NVTEScalingMode = NVTEScalingMode
    stub.CommOverlapType = CommOverlapType
    stub.CommOverlapAlgo = CommOverlapAlgo

    # Register class stubs
    stub.FP8TensorMeta = FP8TensorMeta
    stub.CommOverlapHelper = CommOverlapHelper
    stub.CommOverlap = CommOverlap
    stub.CommOverlapP2P = CommOverlapP2P
    stub.CommOverlapBase = CommOverlapBase
    stub.CommOverlapCore = CommOverlapCore
    stub.CommOverlapP2PBase = CommOverlapP2PBase

    # Register op dispatchers for all registered ops
    for op_name in manager.registry.list_operators():
        setattr(stub, op_name, _make_op_dispatcher(op_name))

    # Ensure all TE main ops have dispatchers even if no backend registered them,
    # so `from transformer_engine_torch import <op>` succeeds at import time.
    # Derived from TEFLBackendBase rather than a second hand-kept list: that class
    # is the plugin's mirror of TE main's pybind surface, so the two cannot drift.
    for op_name in _te_main_op_names():
        if not hasattr(stub, op_name):
            setattr(stub, op_name, _make_op_dispatcher(op_name))

    # Register grouped_mlp_experimental submodule
    gme = types.ModuleType("transformer_engine_torch.grouped_mlp_experimental")
    gme.__doc__ = "Experimental helpers for the fused grouped MLP (unstable, may change or disappear)."
    gme.__package__ = module_name
    gme.swizzle_scales_and_pack_ptrs_for_discrete_weights = _make_op_dispatcher(
        "swizzle_scales_and_pack_ptrs_for_discrete_weights"
    )
    stub.grouped_mlp_experimental = gme
    sys.modules["transformer_engine_torch.grouped_mlp_experimental"] = gme

    sys.modules[module_name] = stub
    logger.info("TE stub module installed (%d ops registered)", len(manager.registry.list_operators()))
