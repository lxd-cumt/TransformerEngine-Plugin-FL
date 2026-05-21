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
    CommOverlapType,
    CommOverlapAlgo,
    FP8TensorMeta,
    CommOverlapHelper,
    CommOverlap,
    CommOverlapP2P,
)

logger = logging.getLogger(__name__)


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
    stub.CommOverlapType = CommOverlapType
    stub.CommOverlapAlgo = CommOverlapAlgo

    # Register class stubs
    stub.FP8TensorMeta = FP8TensorMeta
    stub.CommOverlapHelper = CommOverlapHelper
    stub.CommOverlap = CommOverlap
    stub.CommOverlapP2P = CommOverlapP2P

    # Register op dispatchers for all registered ops
    for op_name in manager.registry.list_operators():
        setattr(stub, op_name, _make_op_dispatcher(op_name))

    sys.modules[module_name] = stub
    logger.info("TE stub module installed (%d ops registered)", len(manager.registry.list_operators()))
