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
    CommOverlapType,
    CommOverlapAlgo,
    FP8TensorMeta,
    CommOverlapHelper,
    CommOverlap,
    CommOverlapP2P,
    CommOverlapBase,
    CommOverlapCore,
    CommOverlapP2PBase,
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
    stub.NVTERoutingMapFormat = NVTERoutingMapFormat
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

    # Ensure all TE main ops have dispatchers even if no backend registered them.
    # This allows `from transformer_engine_torch import <op>` at import time to succeed.
    _ALL_TE_MAIN_OPS = [
        "bgrad_group_quantize", "bgrad_quantize", "bulk_allocate",
        "bulk_overlap_ag_with_external_gemm", "clamped_dswiglu", "clamped_swiglu",
        "compute_amax", "convert_bshd_to_thd", "convert_thd_to_bshd",
        "copy_data_ptrs_to_device", "copy_to_kv_cache",
        "create_empty_quantized_tensor", "cusolvermp_ctx_create",
        "cusolvermp_ctx_destroy", "dbias_dgelu", "dbias_dqgelu", "dbias_drelu",
        "dbias_dsilu", "dbias_dsrelu", "dequantize", "device_supports_multicast",
        "dgeglu", "dgelu", "dglu", "dqgeglu", "dqgelu", "dreglu", "drelu",
        "dropout_bwd", "dropout_fwd", "dsilu", "dsreglu", "dsrelu", "dswiglu",
        "fa_prepare_bwd", "fa_prepare_fwd", "fp8_block_scaling_compute_partial_amax",
        "fp8_block_scaling_partial_cast", "fp8_transpose",
        "fused_amax_and_scale_update_after_reduction", "fused_attn_bwd",
        "fused_attn_fwd", "fused_moe_aux_loss_bwd", "fused_moe_aux_loss_fwd",
        "fused_moe_aux_loss_fwd_graph_safe", "fused_multi_row_padding",
        "fused_multi_row_unpadding", "fused_qkv_rope_backward",
        "fused_qkv_rope_forward", "fused_rope_backward", "fused_rope_forward",
        "fused_score_for_moe_aux_loss_bwd", "fused_score_for_moe_aux_loss_fwd",
        "fused_topk_with_score_function_bwd", "fused_topk_with_score_function_fwd",
        "geglu", "gelu", "generic_gemm", "get_cublasLt_version", "get_cudnn_version",
        "get_fused_attn_backend", "get_grouped_gemm_setup_workspace_size",
        "get_num_cublas_streams", "get_stream_priority_range", "glu",
        "group_dequantize", "group_quantize", "group_requantize_inplace",
        "grouped_swizzle_for_gemm", "init_nvshmem_backend", "layernorm_bwd",
        "layernorm_fwd", "moe_permute_bwd", "moe_permute_fwd", "moe_unpermute_bwd",
        "moe_unpermute_fwd", "multi_tensor_adam", "multi_tensor_adam_capturable",
        "multi_tensor_adam_capturable_master", "multi_tensor_adam_fp8",
        "multi_tensor_adam_param_remainder", "multi_tensor_compute_scale_and_scale_inv",
        "multi_tensor_compute_scale_inv_e8m0", "multi_tensor_l2norm",
        "multi_tensor_pad_last_dim", "multi_tensor_quantize", "multi_tensor_scale",
        "multi_tensor_scale_tensor", "multi_tensor_sgd",
        "multi_tensor_swizzle_scales_for_gemm_",
        "multi_tensor_swizzle_scales_for_gemm_unchecked_",
        "multi_tensor_transpose_to_bhsd", "multi_tensor_unscale_l2norm",
        "mxfp8_scaling_compute_partial_amax", "mxfp8_scaling_partial_cast",
        "newton_schulz", "nvfp4_2d_compute_partial_amax",
        "nvfp4_2d_multi_tensor_transpose", "nvfp4_2d_partial_cast",
        "nvfp4_2d_scale_transpose", "nvfp4_compute_global_scale",
        "nvfp4_compute_per_block_scale", "nvfp4_data_transpose",
        "nvfp4_expand_scale_to_fp8", "nvfp4_fused_scale",
        "nvfp4_group_quantize_with_amax", "nvfp4_multi_tensor_2d_partial_cast",
        "nvfp4_multi_tensor_compute_partial_amax", "nvfp4_multi_tensor_fused_scale",
        "nvfp4_quantize_with_amax", "nvshmem_finalize",
        "nvshmem_send_on_current_stream", "nvshmem_wait_on_current_stream",
        "nvte_built_with_cublasmp", "qgeglu", "qgelu", "quantize", "reglu", "relu",
        "rmsnorm_bwd", "rmsnorm_bwd_add", "rmsnorm_fwd",
        "scaled_aligned_causal_masked_softmax_backward",
        "scaled_aligned_causal_masked_softmax_forward",
        "scaled_clamped_dswiglu", "scaled_clamped_swiglu",
        "scaled_dsrelu", "scaled_dswiglu",
        "scaled_masked_softmax_backward", "scaled_masked_softmax_forward",
        "scaled_softmax_backward", "scaled_softmax_forward",
        "scaled_srelu", "scaled_swiglu",
        "scaled_upper_triang_masked_softmax_backward",
        "scaled_upper_triang_masked_softmax_forward",
        "silu", "split_quantize", "splits_to_offsets", "splits_to_offsets_multi",
        "sreglu", "srelu", "swap_first_dims", "swiglu",
        "swizzle_scales_and_pack_ptrs_for_discrete_weights",
        "swizzle_scales_for_gemm_",
        "te_general_grouped_gemm", "te_general_grouped_gemm_for_discrete_in",
        "te_general_grouped_gemm_for_discrete_out",
        "te_general_grouped_gemm_for_grouped_tensor",
        "thd_copy_valid_tokens_from_per_split_to_rank_local",
        "thd_cp_rank_order_to_sequence_order",
        "thd_get_partitioned_indices", "thd_grad_correction",
        "thd_out_correction", "thd_read_half_tensor", "thd_read_second_half_lse",
        "thd_second_half_lse_correction",
        "thd_sequence_order_to_cp_rank_order",
        "ubuf_built_with_mpi",
    ]
    for op_name in _ALL_TE_MAIN_OPS:
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
