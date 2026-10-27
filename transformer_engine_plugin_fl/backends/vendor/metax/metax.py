# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Any, Dict, List, Optional, Tuple, Union

import ctypes
from pathlib import Path
import importlib.util
import platform
import os
import sys
import functools
import inspect

import torch

from ....ops import *


def _load_metax_libs():

    def get_ext():
        system = platform.system()
        return ".so" if system == "Linux" else ".dylib" if system == "Darwin" else ".dll"

    ext = get_ext()

    # Layout 1 (legacy): MetaX TE <= 2.9 ships as a `transformer_engine_metax`
    # package with the core lib next to it.
    try:
        import transformer_engine_metax

        te_path = Path(importlib.util.find_spec("transformer_engine_metax").origin).parent.parent
        for search_dir in [te_path, te_path / "transformer_engine_metax"]:
            if search_dir.exists():
                matches = list(search_dir.glob(f"libtransformer_engine{ext}*"))
                if matches:
                    ctypes.CDLL(str(matches[0]), mode=ctypes.RTLD_GLOBAL)
                    return True
    except Exception:
        pass

    # Layout 2 (vanilla): MetaX TE >= 2.13 ships as a plain `transformer_engine`
    # wheel. Installing it into site-packages would clobber this (editable)
    # TE-FL package, so it is expected at an isolated prefix pointed to by the
    # TE_METAX_TE_HOME environment variable.
    home = os.environ.get("TE_METAX_TE_HOME", "").strip()
    if home:
        home = Path(home)
        pkg = home / "transformer_engine" if (home / "transformer_engine").is_dir() else home
        for search_dir in [pkg, home]:
            if search_dir.exists():
                matches = sorted(search_dir.glob(f"libtransformer_engine{ext}*"))
                if matches:
                    ctypes.CDLL(str(matches[0]), mode=ctypes.RTLD_GLOBAL)
                    return True
    return False


_metax_libs_loaded = False


def _ensure_metax_libs():
    global _metax_libs_loaded
    if not _metax_libs_loaded:
        _metax_libs_loaded = _load_metax_libs()
        if _metax_libs_loaded:
            print(f"[Metax] Successfully loaded Metax libs")
    return _metax_libs_loaded


def _import_tex_module():
    """Import the Metax torch extension robustly, both package layouts.

    - Legacy: ``transformer_engine_torch_metax`` importable next to the
      ``transformer_engine_metax`` package (MetaX TE 2.9 era).
    - Vanilla: MetaX TE >= 2.13 ships the extension as
      ``transformer_engine/transformer_engine_torch*.so`` inside a plain
      ``transformer_engine`` wheel. It is loaded by file path from the
      isolated prefix in ``TE_METAX_TE_HOME`` and aliased to
      ``transformer_engine_torch_metax`` in ``sys.modules``.

    Extensions built against torch<=2.8 headers reference
    ``c10::SymInt::sym_ne``, which torch>=2.10 no longer exports (it became
    header-inline). Loading under RTLD_LAZY defers that dangling binding; it
    is harmless on call paths that never touch it.
    """
    mod = sys.modules.get("transformer_engine_torch_metax")
    if mod is not None:
        return mod

    if importlib.util.find_spec("transformer_engine_metax") is not None:
        import transformer_engine_torch_metax

        return transformer_engine_torch_metax

    home = os.environ.get("TE_METAX_TE_HOME", "").strip()
    if not home:
        raise ImportError(
            "transformer_engine_torch_metax not found and TE_METAX_TE_HOME is not set"
        )
    home = Path(home)
    pkg = home / "transformer_engine" if (home / "transformer_engine").is_dir() else home
    matches = sorted(pkg.glob("transformer_engine_torch*.so"))
    if not matches:
        raise ImportError(f"no transformer_engine_torch*.so found under {pkg}")

    old_flags = sys.getdlopenflags()
    # Single-phase extension init registers itself in sys.modules under its
    # PyInit name ("transformer_engine_torch"), which would shadow the TEFL
    # dispatch module that plugin.core._module_setup installs under the same
    # name. Save the current owner and restore it right after exec_module.
    saved_owner = sys.modules.pop("transformer_engine_torch", None)
    try:
        sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)
        spec = importlib.util.spec_from_file_location("transformer_engine_torch", str(matches[0]))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.setdlopenflags(old_flags)
        if sys.modules.get("transformer_engine_torch") is mod:
            if saved_owner is not None:
                sys.modules["transformer_engine_torch"] = saved_owner
            else:
                del sys.modules["transformer_engine_torch"]
    sys.modules["transformer_engine_torch_metax"] = mod
    return mod


def _check_metax_available() -> bool:
    if not torch.cuda.is_available():
        return False

    try:
        from ...._build_config import SKIP_METAX_BUILD

        if SKIP_METAX_BUILD:
            print("[Metax] Disabled: Metax was skipped at build time")
            return False
    except ImportError:
        if bool(int(os.environ.get("TE_FL_SKIP_METAX", "0"))):
            print("[Metax] Disabled: TE_FL_SKIP_METAX=1")
            return False

    try:
        if not _ensure_metax_libs():
            return False
        _import_tex_module()

        return True
    except (ImportError, OSError) as e:
        print(f"[Metax] Import failed: {e}")
        return False


def _get_tex():
    _ensure_metax_libs()
    return _import_tex_module()


class MetaxBackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return _check_metax_available()

    def __init__(self):
        self._tex = None

    def _get_tex(self):
        if self._tex is None:
            self._tex = vendor_te_compat(_get_tex(), "MetaX")
        return self._tex

    def _normalize_quantizer_dtype(self, quantizer: Any) -> Any:
        """Re-wrap ``quantizer.dtype`` as this backend's pybind ``tex.DType``.

        Quantizers built against the plugin's pure-Python ``DType`` carry an
        IntEnum the pybind layer will not accept, so convert by integer value.
        """
        try:
            tex = self._get_tex()
            if quantizer is not None and hasattr(quantizer, "dtype") and hasattr(tex, "DType"):
                qdtype = quantizer.dtype
                if qdtype is not None:
                    quantizer.dtype = tex.DType(int(qdtype))
        except Exception:
            pass
        return quantizer

    def is_available(self) -> bool:
        return _check_metax_available()

    def get_attention_backend(self, attention_params=None):
        # Import the metax get_attention_backend function
        try:
            from transformer_engine_metax.pytorch.attention.dot_product_attention import utils

            return utils.get_attention_backend(attention_params)

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import metax FlashAttention: {e}. "
                "Please ensure flash-attn is installed and transformer_engine_metax is available."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get_attention_backend: {e}. Attention_params: {self.attention_params}"
            )

    def get_flash_attention_class(self):
        from .flash_attention import FlashAttentionMETAX

        return FlashAttentionMETAX

    def create_comm_overlap(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        helper: Any,
        tp_size: int,
        use_cublasmp: bool = False,
        comm_type: Any = None,
        num_splits: int = 4,
        num_max_streams: int = 3,
        comm_cga_size: int = 2,
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = 16,
        set_sm_margin: bool = True,
        atomic_gemm: bool = False,
        rs_overlap_first_gemm: bool = False,
    ) -> "CommOverlap":
        if use_cublasmp:
            raise NotImplementedError("metax does not support cuBLASMp overlap")
        # The vendor constructor has no comm_type argument; upstream only uses
        # it on the cuBLASMp path, which is rejected above.
        tex = self._get_tex()
        return tex.CommOverlap(
            buffer_shape,
            buffer_dtype,
            helper,
            tp_size,
            num_splits,
            num_max_streams,
            comm_cga_size,
            gemm_priority,
            comm_priority,
            num_comm_sm,
            set_sm_margin,
            atomic_gemm,
            rs_overlap_first_gemm,
        )

    def create_comm_overlap_p2p(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        helper: Any,
        tp_size: int,
        comm_type: Any,
        num_max_streams: int = 3,
        comm_cga_size: int = 1,
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = 1,
        set_sm_margin: bool = False,
        atomic_gemm: bool = False,
        use_ce: bool = True,
        aggregate: bool = False,
        use_cublasmp: bool = False,
    ) -> "CommOverlapP2P":
        if use_cublasmp:
            raise NotImplementedError("metax does not support cuBLASMp overlap")
        tex = self._get_tex()
        return tex.CommOverlapP2P(
            buffer_shape,
            buffer_dtype,
            helper,
            tp_size,
            comm_type,
            num_max_streams,
            comm_cga_size,
            gemm_priority,
            comm_priority,
            num_comm_sm,
            set_sm_margin,
            atomic_gemm,
            use_ce,
            aggregate,
        )

    def multi_tensor_scale_tensor(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: torch.Tensor,
    ) -> None:
        # transformer_engine_torch_metax does not support multi_tensor_scale_tensor
        # (from upstream Nvidia TE v2.14). Use multi_tensor_scale as a workaround.
        tex = self._get_tex()
        scale_value = scale.item()
        return tex.multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale_value)

    ##### BEGIN GENERATED OPS (tools/gen_vendor_ops.py) #####
    def quantize(
        self,
        tensor: torch.Tensor,
        quantizer: Any,
        output: Optional[torch.Tensor] = None,
        noop: Optional[torch.Tensor] = None,
    ) -> Any:
        tex = self._get_tex()
        self._normalize_quantizer_dtype(quantizer)
        return tex.quantize(tensor, quantizer, output, noop)

    def dequantize(
        self,
        input: Any,
        otype: DType,
    ) -> Any:
        tex = self._get_tex()
        otype = tex.DType(int(otype)) if otype is not None else None
        return tex.dequantize(input, otype)

    def bgrad_quantize(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> List[Any]:
        tex = self._get_tex()
        self._normalize_quantizer_dtype(quantizer)
        return tex.bgrad_quantize(input, quantizer)

    def group_quantize(
        self,
        tensor: torch.Tensor,
        quantizer: Any,
        num_tensors: int,
        first_dims: List[int],
        last_dims: Optional[List[int]] = None,
        tensor_offsets: Optional[torch.Tensor] = None,
        noop_flag: Optional[torch.Tensor] = None,
        output: Optional[Any] = None,
    ) -> Any:
        tex = self._get_tex()
        self._normalize_quantizer_dtype(quantizer)
        return tex.group_quantize(
            tensor,
            quantizer,
            num_tensors,
            first_dims,
            last_dims,
            tensor_offsets,
            noop_flag,
            output,
        )

    def bgrad_group_quantize(
        self,
        tensor: torch.Tensor,
        quantizer: Any,
        num_tensors: int,
        first_dims: List[int],
        last_dims: Optional[List[int]] = None,
        tensor_offsets: Optional[torch.Tensor] = None,
    ) -> Any:
        tex = self._get_tex()
        self._normalize_quantizer_dtype(quantizer)
        return tex.bgrad_group_quantize(
            tensor, quantizer, num_tensors, first_dims, last_dims, tensor_offsets
        )

    def group_dequantize(
        self,
        input: Any,
        otype: DType,
    ) -> Any:
        tex = self._get_tex()
        otype = tex.DType(int(otype)) if otype is not None else None
        return tex.group_dequantize(input, otype)

    def group_requantize_inplace(
        self,
        grouped_x: Any,
        quantizer: Any,
        num_tensors: int,
        first_dims: List[int],
        otype: DType,
        tensor_offsets: Optional[torch.Tensor] = None,
        return_dequantized: bool = False,
    ) -> Any:
        tex = self._get_tex()
        self._normalize_quantizer_dtype(quantizer)
        otype = tex.DType(int(otype)) if otype is not None else None
        return tex.group_requantize_inplace(
            grouped_x,
            quantizer,
            num_tensors,
            first_dims,
            otype,
            tensor_offsets,
            return_dequantized,
        )

    def create_empty_quantized_tensor(
        self,
        quantizer: Any,
        shape: List[int],
        dtype: torch.dtype,
        device: Any,
        pin_memory: bool,
    ) -> Any:
        tex = self._get_tex()
        self._normalize_quantizer_dtype(quantizer)
        return tex.create_empty_quantized_tensor(quantizer, shape, dtype, device, pin_memory)

    def nvfp4_quantize_with_amax(
        self,
        tensor: torch.Tensor,
        quantizer: Any,
        rowwise_amax: Optional[torch.Tensor],
        columnwise_amax: Optional[torch.Tensor],
    ) -> Any:
        tex = self._get_tex()
        self._normalize_quantizer_dtype(quantizer)
        return tex.nvfp4_quantize_with_amax(tensor, quantizer, rowwise_amax, columnwise_amax)

    def nvfp4_group_quantize_with_amax(
        self,
        tensor: torch.Tensor,
        quantizer: Any,
        num_tensors: int,
        first_dims: List[int],
        last_dims: Optional[List[int]] = None,
        rowwise_amax: Optional[torch.Tensor] = None,
        columnwise_amax: Optional[torch.Tensor] = None,
        tensor_offsets: Optional[torch.Tensor] = None,
    ) -> Any:
        tex = self._get_tex()
        self._normalize_quantizer_dtype(quantizer)
        return tex.nvfp4_group_quantize_with_amax(
            tensor,
            quantizer,
            num_tensors,
            first_dims,
            last_dims,
            rowwise_amax=rowwise_amax,
            columnwise_amax=columnwise_amax,
            tensor_offsets=tensor_offsets,
        )

    def nvfp4_gemm_swizzled_scales_alignment(
        self,
        optimize_for_gemm: bool,
        with_rht: bool,
        with_2d_quantization: bool,
        row_scaled_nvfp4: bool,
        nvfp4_use_4over6: bool,
        sm_arch: int,
        disable_rht_cast_fusion: bool,
    ) -> Any:
        tex = self._get_tex()
        return tex.nvfp4_gemm_swizzled_scales_alignment(
            optimize_for_gemm,
            with_rht,
            with_2d_quantization,
            row_scaled_nvfp4,
            nvfp4_use_4over6,
            sm_arch,
            disable_rht_cast_fusion,
        )

    def generic_gemm(
        self,
        A: Any,
        transA: bool,
        B: Any,
        transB: bool,
        D: Any,
        quantizer: Any,
        output_dtype: Optional[DType],
        bias: Optional[torch.Tensor],
        bias_type: DType,
        gelu: bool,
        gelu_in: Optional[torch.Tensor],
        grad: bool,
        workspace: torch.Tensor,
        workspace_size: int,
        accumulate: bool,
        use_split_accumulator: bool,
        comm_overlap: Optional[Any] = None,
        comm_type: Optional[CommOverlapType] = None,
        extra_output: Optional[torch.Tensor] = None,
        bulk_overlap: bool = False,
        alpha: float = 1.0,
        beta: Optional[float] = None,
    ) -> List[Any]:
        tex = self._get_tex()

        bias_type = tex.DType(int(bias_type)) if bias_type is not None else None
        comm_type = tex.CommOverlapType(int(comm_type)) if comm_type is not None else None
        output_dtype = tex.DType(int(output_dtype)) if output_dtype is not None else None
        return tex.generic_gemm(
            A,
            transA,
            B,
            transB,
            D,
            quantizer,
            output_dtype,
            bias,
            bias_type,
            gelu,
            gelu_in,
            grad,
            workspace,
            workspace_size,
            accumulate,
            use_split_accumulator,
            comm_overlap,
            comm_type,
            extra_output,
            bulk_overlap,
            alpha,
            beta,
        )

    def glu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.glu(input, quantizer)

    def gelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.gelu(input, quantizer)

    def geglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.geglu(input, quantizer)

    def qgelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.qgelu(input, quantizer)

    def qgeglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.qgeglu(input, quantizer)

    def relu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.relu(input, quantizer)

    def reglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.reglu(input, quantizer)

    def srelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.srelu(input, quantizer)

    def sreglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.sreglu(input, quantizer)

    def silu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.silu(input, quantizer)

    def swiglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.swiglu(input, quantizer)

    def clamped_swiglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.702,
        glu_linear_offset: float = 1.0,
    ) -> Any:
        tex = self._get_tex()
        return tex.clamped_swiglu(input, quantizer, limit, alpha, glu_linear_offset)

    def situglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
        beta1: float = 4.0,
        beta2: float = 25.0,
    ) -> Any:
        tex = self._get_tex()
        return tex.situglu(input, quantizer, beta1, beta2)

    def dglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dglu(grad, fwd_input, quantizer)

    def dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dgelu(grad, fwd_input, quantizer)

    def dgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dgeglu(grad, fwd_input, quantizer)

    def dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dqgelu(grad, fwd_input, quantizer)

    def dqgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dqgeglu(grad, fwd_input, quantizer)

    def drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.drelu(grad, fwd_input, quantizer)

    def dreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dreglu(grad, fwd_input, quantizer)

    def dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dsrelu(grad, fwd_input, quantizer)

    def dsreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dsreglu(grad, fwd_input, quantizer)

    def dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dsilu(grad, fwd_input, quantizer)

    def dswiglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dswiglu(grad, fwd_input, quantizer)

    def clamped_dswiglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.702,
        glu_linear_offset: float = 1.0,
    ) -> Any:
        tex = self._get_tex()
        return tex.clamped_dswiglu(grad, fwd_input, quantizer, limit, alpha, glu_linear_offset)

    def dsituglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
        beta1: float = 4.0,
        beta2: float = 25.0,
    ) -> Any:
        tex = self._get_tex()
        return tex.dsituglu(grad, fwd_input, quantizer, beta1, beta2)

    def dbias_dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> List[Any]:
        tex = self._get_tex()
        return tex.dbias_dgelu(grad, fwd_input, quantizer)

    def dbias_dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> List[Any]:
        tex = self._get_tex()
        return tex.dbias_dsilu(grad, fwd_input, quantizer)

    def dbias_drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> List[Any]:
        tex = self._get_tex()
        return tex.dbias_drelu(grad, fwd_input, quantizer)

    def dbias_dqgelu(
        self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.dbias_dqgelu(grad, fwd_input, quantizer)

    def dbias_dsrelu(
        self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.dbias_dsrelu(grad, fwd_input, quantizer)

    def moe_permute_fwd(
        self,
        input: torch.Tensor,
        dtype: DType,
        indices: torch.Tensor,
        num_out_tokens: int,
        workspace: List[torch.Tensor],
        max_expanded_token_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        tex = self._get_tex()
        dtype = tex.DType(int(dtype)) if dtype is not None else None
        return tex.moe_permute_fwd(
            input, dtype, indices, num_out_tokens, workspace, max_expanded_token_num
        )

    def moe_permute_bwd(
        self,
        input: torch.Tensor,
        dtype: DType,
        row_id_map: torch.Tensor,
        prob: torch.Tensor,
        num_tokens: int,
        topK: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        dtype = tex.DType(int(dtype)) if dtype is not None else None
        return tex.moe_permute_bwd(input, dtype, row_id_map, prob, num_tokens, topK)

    def moe_unpermute_fwd(
        self,
        input: torch.Tensor,
        dtype: DType,
        row_id_map: torch.Tensor,
        prob: torch.Tensor,
        num_tokens: int,
        topK: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        dtype = tex.DType(int(dtype)) if dtype is not None else None
        return tex.moe_unpermute_fwd(input, dtype, row_id_map, prob, num_tokens, topK)

    def moe_unpermute_bwd(
        self,
        input_bwd: torch.Tensor,
        input_fwd: torch.Tensor,
        dtype: DType,
        row_id_map: torch.Tensor,
        prob: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        dtype = tex.DType(int(dtype)) if dtype is not None else None
        return tex.moe_unpermute_bwd(input_bwd, input_fwd, dtype, row_id_map, prob)

    def scaled_softmax_forward(
        self,
        input: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_softmax_forward(input, scale_factor)

    def scaled_softmax_backward(
        self,
        output_grad_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_softmax_backward(output_grad_, softmax_results_, scale_factor)

    def scaled_masked_softmax_forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_masked_softmax_forward(input, mask, scale_factor)

    def scaled_masked_softmax_backward(
        self,
        output_grad_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_masked_softmax_backward(output_grad_, softmax_results_, scale_factor)

    def scaled_upper_triang_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_upper_triang_masked_softmax_forward(input, scale_factor)

    def scaled_upper_triang_masked_softmax_backward(
        self,
        output_grads_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_upper_triang_masked_softmax_backward(
            output_grads_, softmax_results_, scale_factor
        )

    def scaled_aligned_causal_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_aligned_causal_masked_softmax_forward(input, scale_factor)

    def scaled_aligned_causal_masked_softmax_backward(
        self,
        output_grads_: torch.Tensor,
        softmax_results_: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_aligned_causal_masked_softmax_backward(
            output_grads_, softmax_results_, scale_factor
        )

    def layernorm_fwd(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
        ln_out: Any,
        quantizer: Any,
        otype: DType,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> List[Any]:
        tex = self._get_tex()
        otype = tex.DType(int(otype)) if otype is not None else None
        return tex.layernorm_fwd(
            input,
            weight,
            bias,
            eps,
            ln_out,
            quantizer,
            otype,
            sm_margin,
            zero_centered_gamma,
        )

    def layernorm_bwd(
        self,
        dz: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.layernorm_bwd(dz, x, mu, rsigma, gamma, sm_margin, zero_centered_gamma)

    def rmsnorm_fwd(
        self,
        input: Any,
        weight: Any,
        eps: float,
        ln_out: Any,
        quantizer: Any,
        otype: DType,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> List[Any]:
        tex = self._get_tex()
        otype = tex.DType(int(otype)) if otype is not None else None
        return tex.rmsnorm_fwd(
            input, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma
        )

    def rmsnorm_bwd(
        self,
        dz: torch.Tensor,
        x: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)

    def rmsnorm_bwd_add(
        self,
        dz: torch.Tensor,
        x: torch.Tensor,
        add: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.rmsnorm_bwd_add(dz, x, add, rsigma, gamma, sm_margin, zero_centered_gamma)

    def multi_tensor_quantize(
        self,
        tensor_list: List[torch.Tensor],
        quantizer_list: List[Any],
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.multi_tensor_quantize(tensor_list, quantizer_list)

    def split_quantize(
        self,
        tensor: torch.Tensor,
        split_sections: List[int],
        quantizer_list: List[Any],
        disable_bulk_allocation: bool = False,
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.split_quantize(tensor, split_sections, quantizer_list, disable_bulk_allocation)

    def te_general_grouped_gemm(
        self,
        A: List[Any],
        transa: bool,
        B: List[Any],
        transb: bool,
        D: Optional[List[torch.Tensor]],
        D_type: DType,
        m_splits: List[int],
        bias: List[torch.Tensor],
        bias_type: DType,
        single_output: bool,
        pre_gelu_out: List[torch.Tensor],
        grad: bool,
        workspace: List[torch.Tensor],
        workspaceSize: int,
        accumulate: bool,
        use_split_accumulator: bool,
        math_sm_count: int,
    ) -> Optional[List[torch.Tensor]]:
        tex = self._get_tex()
        D_type = tex.DType(int(D_type)) if D_type is not None else None
        bias_type = tex.DType(int(bias_type)) if bias_type is not None else None
        return tex.te_general_grouped_gemm(
            A,
            transa,
            B,
            transb,
            D,
            D_type,
            m_splits,
            bias,
            bias_type,
            single_output,
            pre_gelu_out,
            grad,
            workspace,
            workspaceSize,
            accumulate,
            use_split_accumulator,
            math_sm_count,
        )

    def te_general_grouped_gemm_for_grouped_tensor(self, *args, **kwargs):
        tex = self._get_tex()
        return tex.te_general_grouped_gemm_for_grouped_tensor(*args, **kwargs)

    def te_general_grouped_gemm_for_discrete_in(self, *args, **kwargs):
        tex = self._get_tex()
        return tex.te_general_grouped_gemm_for_discrete_in(*args, **kwargs)

    def te_general_grouped_gemm_for_discrete_out(self, *args, **kwargs):
        tex = self._get_tex()
        return tex.te_general_grouped_gemm_for_discrete_out(*args, **kwargs)

    def fp8_transpose(
        self,
        input: torch.Tensor,
        dtype: DType,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tex = self._get_tex()
        dtype = tex.DType(int(dtype)) if dtype is not None else None
        return tex.fp8_transpose(input, dtype, out=out)

    def swap_first_dims(
        self,
        tensor: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.swap_first_dims(tensor, out=out)

    def nvfp4_data_transpose(
        self,
        input: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.nvfp4_data_transpose(input, out=out)

    def swizzle_scales_for_gemm_(self, tensor: torch.Tensor) -> None:
        tex = self._get_tex()
        return tex.swizzle_scales_for_gemm_(tensor)

    def grouped_swizzle_for_gemm(
        self,
        tensor: Any,
        rowwise: bool,
        columnwise: bool,
    ) -> None:
        tex = self._get_tex()
        return tex.grouped_swizzle_for_gemm(tensor, rowwise, columnwise)

    def convert_host_pointers_to_tensor(
        self,
        tensor_lists: List[List[torch.Tensor]],
    ) -> Any:
        tex = self._get_tex()
        return tex.convert_host_pointers_to_tensor(tensor_lists)

    def get_device_pointer_for_data_and_scales(
        self,
        data_tensors: List[torch.Tensor],
        scale_tensors: List[torch.Tensor],
        swizzle: bool = False,
        rowwise: bool = True,
        data_dtype: Any = None,
    ) -> Any:
        tex = self._get_tex()
        return tex.get_device_pointer_for_data_and_scales(
            data_tensors, scale_tensors, swizzle, rowwise, data_dtype
        )

    def splits_to_offsets(
        self,
        first_dims: List[int],
        logical_last_dim: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.splits_to_offsets(first_dims, logical_last_dim)

    def get_fused_attn_backend(
        self,
        fused_attn_params: Any,
    ) -> Tuple[NVTE_Fused_Attn_Backend, str]:
        tex = self._get_tex()

        for attr, enum_name in self._FUSED_ATTN_PARAM_ENUMS.items():
            value = getattr(fused_attn_params, attr, None)
            if value is None:
                continue
            tex_enum = getattr(tex, enum_name, None)
            if tex_enum is None:
                continue
            try:
                setattr(fused_attn_params, attr, tex_enum(int(value)))
            except Exception:
                pass

        backend, message = tex.get_fused_attn_backend(fused_attn_params)
        return NVTE_Fused_Attn_Backend(int(backend)), message

    def compute_amax(
        self,
        input: torch.Tensor,
        amax: torch.Tensor,
    ) -> None:
        tex = self._get_tex()
        return tex.compute_amax(input, amax)

    def fused_amax_and_scale_update_after_reduction(
        self,
        amax_reduction_buffer: torch.Tensor,
        amax_histories: List[torch.Tensor],
        scales: List[torch.Tensor],
        amax_compute_algo: str,
        fp8_dtype: DType,
        margin: float,
    ) -> None:
        tex = self._get_tex()
        fp8_dtype = tex.DType(int(fp8_dtype)) if fp8_dtype is not None else None
        return tex.fused_amax_and_scale_update_after_reduction(
            amax_reduction_buffer,
            amax_histories,
            scales,
            amax_compute_algo,
            fp8_dtype,
            margin,
        )

    def fp8_block_scaling_compute_partial_amax(
        self,
        tensor: torch.Tensor,
        amax: torch.Tensor,
        h: int,
        w: int,
        start_offset: int,
        block_len: int,
    ) -> None:
        tex = self._get_tex()
        return tex.fp8_block_scaling_compute_partial_amax(
            tensor, amax, h, w, start_offset, block_len
        )

    def fp8_block_scaling_partial_cast(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        scale: torch.Tensor,
        h: int,
        w: int,
        start_offset: int,
        block_len: int,
        out_dtype: DType,
    ) -> None:
        tex = self._get_tex()
        out_dtype = tex.DType(int(out_dtype)) if out_dtype is not None else None
        return tex.fp8_block_scaling_partial_cast(
            inp, out, scale, h, w, start_offset, block_len, out_dtype
        )

    def mxfp8_scaling_compute_partial_amax(
        self,
        input: torch.Tensor,
        amax_rowwise: torch.Tensor,
        amax_colwise: torch.Tensor,
        rows: int,
        cols: int,
        start_offset: int,
    ) -> None:
        tex = self._get_tex()
        return tex.mxfp8_scaling_compute_partial_amax(
            input, amax_rowwise, amax_colwise, rows, cols, start_offset
        )

    def mxfp8_scaling_partial_cast(
        self,
        input: torch.Tensor,
        output_rowwise: torch.Tensor,
        output_colwise: torch.Tensor,
        scale_inv_rowwise: torch.Tensor,
        scale_inv_colwise: torch.Tensor,
        rows: int,
        cols: int,
        start_offset: int,
    ) -> None:
        tex = self._get_tex()
        return tex.mxfp8_scaling_partial_cast(
            input,
            output_rowwise,
            output_colwise,
            scale_inv_rowwise,
            scale_inv_colwise,
            rows,
            cols,
            start_offset,
        )

    def nvfp4_2d_compute_partial_amax(
        self,
        tensor: torch.Tensor,
        amax: torch.Tensor,
        h: int,
        w: int,
        start_offset: int,
        block_len: int = 16,
    ) -> None:
        tex = self._get_tex()
        return tex.nvfp4_2d_compute_partial_amax(tensor, amax, h, w, start_offset, block_len)

    def nvfp4_multi_tensor_compute_partial_amax(
        self,
        master_weight_list: List[torch.Tensor],
        partial_amax_list: List[torch.Tensor],
        global_amax_list: List[torch.Tensor],
        h_list: List[int],
        w_list: List[int],
        start_offset_list: List[int],
        block_len: int = 16,
    ) -> None:
        tex = self._get_tex()
        return tex.nvfp4_multi_tensor_compute_partial_amax(
            master_weight_list,
            partial_amax_list,
            global_amax_list,
            h_list,
            w_list,
            start_offset_list,
            block_len,
        )

    def nvfp4_compute_global_scale(
        self,
        global_amax: torch.Tensor,
        global_scale: torch.Tensor,
        scale_dtype: DType = DType.kFloat8E4M3,
    ) -> None:
        tex = self._get_tex()
        return tex.nvfp4_compute_global_scale(
            global_amax, global_scale, tex.DType(int(scale_dtype))
        )

    def nvfp4_compute_per_block_scale(
        self,
        block_amax: torch.Tensor,
        scale: torch.Tensor,
        global_amax: torch.Tensor,
        scale_dtype: DType = DType.kFloat8E4M3,
    ) -> None:
        tex = self._get_tex()
        return tex.nvfp4_compute_per_block_scale(
            block_amax, scale, global_amax, tex.DType(int(scale_dtype))
        )

    def nvfp4_expand_scale_to_fp8(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        tile_rows: int,
        tile_cols: int,
        rows_padded: int,
        block_len: int,
        scale_dtype: DType = DType.kFloat8E4M3,
    ) -> None:
        tex = self._get_tex()
        return tex.nvfp4_expand_scale_to_fp8(
            input,
            output,
            tile_rows,
            tile_cols,
            rows_padded,
            block_len,
            tex.DType(int(scale_dtype)),
        )

    def nvfp4_fused_scale(
        self,
        block_amax: torch.Tensor,
        global_amax: torch.Tensor,
        per_block_scale: torch.Tensor,
        target_scale: torch.Tensor,
        target_amax: torch.Tensor,
        tile_rows: int,
        tile_cols: int,
        rows_padded: int,
        block_len: int,
        scale_dtype: DType = DType.kFloat8E4M3,
    ) -> None:
        tex = self._get_tex()
        return tex.nvfp4_fused_scale(
            block_amax,
            global_amax,
            per_block_scale,
            target_scale,
            target_amax,
            tile_rows,
            tile_cols,
            rows_padded,
            block_len,
            tex.DType(int(scale_dtype)),
        )

    def nvfp4_multi_tensor_fused_scale(
        self,
        block_amax_list: List[torch.Tensor],
        global_amax_list: List[torch.Tensor],
        per_block_scale_list: List[torch.Tensor],
        target_scale_list: List[torch.Tensor],
        target_amax_list: List[torch.Tensor],
        tile_rows_list: List[int],
        tile_cols_list: List[int],
        rows_padded_list: List[int],
        block_len: int,
        scale_dtype: DType = DType.kFloat8E4M3,
    ) -> None:
        tex = self._get_tex()
        return tex.nvfp4_multi_tensor_fused_scale(
            block_amax_list,
            global_amax_list,
            per_block_scale_list,
            target_scale_list,
            target_amax_list,
            tile_rows_list,
            tile_cols_list,
            rows_padded_list,
            block_len,
            tex.DType(int(scale_dtype)),
        )

    def nvfp4_2d_partial_cast(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        scale: torch.Tensor,
        global_scale: torch.Tensor,
        h: int,
        w: int,
        start_offset: int,
        block_len: int = 16,
        scale_dtype: DType = DType.kFloat8E4M3,
    ) -> None:
        tex = self._get_tex()
        return tex.nvfp4_2d_partial_cast(
            inp,
            out,
            scale,
            global_scale,
            h,
            w,
            start_offset,
            block_len,
            tex.DType(int(scale_dtype)),
        )

    def nvfp4_multi_tensor_2d_partial_cast(
        self,
        inp_list: List[torch.Tensor],
        out_list: List[torch.Tensor],
        scale_list: List[torch.Tensor],
        global_scale_list: List[torch.Tensor],
        h_list: List[int],
        w_list: List[int],
        start_offset_list: List[int],
        block_len: int = 16,
        scale_dtype: DType = DType.kFloat8E4M3,
    ) -> None:
        tex = self._get_tex()
        return tex.nvfp4_multi_tensor_2d_partial_cast(
            inp_list,
            out_list,
            scale_list,
            global_scale_list,
            h_list,
            w_list,
            start_offset_list,
            block_len,
            tex.DType(int(scale_dtype)),
        )

    def nvfp4_2d_scale_transpose(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        M_tiles: int,
        K_tiles: int,
    ) -> None:
        tex = self._get_tex()
        return tex.nvfp4_2d_scale_transpose(input, output, M_tiles, K_tiles)

    def nvfp4_2d_multi_tensor_transpose(
        self,
        rowwise_data_list: List[torch.Tensor],
        columnwise_data_list: List[torch.Tensor],
        rowwise_scale_inv_list: List[torch.Tensor],
        columnwise_scale_inv_list: List[torch.Tensor],
        M_list: List[int],
        K_list: List[int],
    ) -> None:
        tex = self._get_tex()
        return tex.nvfp4_2d_multi_tensor_transpose(
            rowwise_data_list,
            columnwise_data_list,
            rowwise_scale_inv_list,
            columnwise_scale_inv_list,
            M_list,
            K_list,
        )

    def fused_multi_row_padding(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        input_row_list: List[int],
        padded_input_row_list: List[int],
    ) -> None:
        tex = self._get_tex()
        return tex.fused_multi_row_padding(input, output, input_row_list, padded_input_row_list)

    def fused_multi_row_unpadding(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        input_row_list: List[int],
        unpadded_input_row_list: List[int],
    ) -> None:
        tex = self._get_tex()
        return tex.fused_multi_row_unpadding(input, output, input_row_list, unpadded_input_row_list)

    def fa_prepare_fwd(
        self,
        qkvi: torch.Tensor,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fa_prepare_fwd(qkvi)

    def fa_prepare_bwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fa_prepare_bwd(q, k, v)

    def fused_attn_fwd(
        self,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        is_training: bool,
        attn_scale: float,
        p_dropout: float,
        set_zero: bool,
        qkv_layout: NVTE_QKV_Layout,
        o_format: NVTE_QKV_Format,
        qkv_scale_inv_format: NVTE_QKV_Format,
        bias_type: NVTE_Bias_Type,
        attn_mask_type: NVTE_Mask_Type,
        softmax_type: NVTE_Softmax_Type,
        window_size: List[int],
        bottom_right_diagonal: bool,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        Q: Any,
        K: Any,
        V: Any,
        fake_dtype: torch.dtype,
        cu_seqlens_q_padded: Optional[torch.Tensor],
        cu_seqlens_kv_padded: Optional[torch.Tensor],
        page_table_k: Optional[torch.Tensor],
        page_table_v: Optional[torch.Tensor],
        s_quantizer: Any,
        o_quantizer: Any,
        Bias: Optional[torch.Tensor],
        SoftmaxOffset: Optional[torch.Tensor],
        rng_gen: Optional[torch.Generator],
        rng_elts_per_thread: int,
        return_max_logit: bool,
        cuda_graph: bool = False,
    ) -> List[Any]:
        tex = self._get_tex()

        qkv_layout = tex.NVTE_QKV_Layout(int(qkv_layout)) if qkv_layout is not None else None
        bias_type = tex.NVTE_Bias_Type(int(bias_type)) if bias_type is not None else None
        attn_mask_type = (
            tex.NVTE_Mask_Type(int(attn_mask_type)) if attn_mask_type is not None else None
        )
        softmax_type = (
            tex.NVTE_Softmax_Type(int(softmax_type)) if softmax_type is not None else None
        )
        o_format = tex.NVTE_QKV_Format(int(o_format)) if o_format is not None else None
        qkv_scale_inv_format = (
            tex.NVTE_QKV_Format(int(qkv_scale_inv_format))
            if qkv_scale_inv_format is not None
            else None
        )

        return tex.fused_attn_fwd(
            max_seqlen_q,
            max_seqlen_kv,
            is_training,
            attn_scale,
            p_dropout,
            set_zero,
            qkv_layout,
            o_format,
            qkv_scale_inv_format,
            bias_type,
            attn_mask_type,
            softmax_type,
            window_size,
            bottom_right_diagonal,
            cu_seqlens_q,
            cu_seqlens_kv,
            Q,
            K,
            V,
            fake_dtype,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            page_table_k,
            page_table_v,
            s_quantizer,
            o_quantizer,
            Bias,
            SoftmaxOffset,
            rng_gen,
            rng_elts_per_thread,
            return_max_logit,
            cuda_graph,
        )

    def fused_attn_bwd(
        self,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        attn_scale: float,
        p_dropout: float,
        set_zero: bool,
        qkv_layout: NVTE_QKV_Layout,
        o_format: NVTE_QKV_Format,
        do_format: NVTE_QKV_Format,
        dqkv_layout: NVTE_QKV_Layout,
        qkv_scale_inv_format: NVTE_QKV_Format,
        do_scale_inv_format: NVTE_QKV_Format,
        bias_type: NVTE_Bias_Type,
        attn_mask_type: NVTE_Mask_Type,
        softmax_type: NVTE_Softmax_Type,
        window_size: List[int],
        bottom_right_diagonal: bool,
        deterministic: bool,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        Q: Any,
        K: Any,
        V: Any,
        O: Any,
        dO: Any,
        fake_dtype: torch.dtype,
        Aux_CTX_Tensors: List[torch.Tensor],
        cu_seqlens_q_padded: Optional[torch.Tensor],
        cu_seqlens_kv_padded: Optional[torch.Tensor],
        s_quantizer: Any,
        dp_quantizer: Any,
        dqkv_quantizer: Any,
        cuda_graph: bool = False,
    ) -> List[Any]:
        tex = self._get_tex()

        qkv_layout = tex.NVTE_QKV_Layout(int(qkv_layout)) if qkv_layout is not None else None
        bias_type = tex.NVTE_Bias_Type(int(bias_type)) if bias_type is not None else None
        attn_mask_type = (
            tex.NVTE_Mask_Type(int(attn_mask_type)) if attn_mask_type is not None else None
        )
        softmax_type = (
            tex.NVTE_Softmax_Type(int(softmax_type)) if softmax_type is not None else None
        )
        dqkv_layout = (
            tex.NVTE_QKV_Layout(int(dqkv_layout)) if dqkv_layout is not None else None
        )
        o_format = tex.NVTE_QKV_Format(int(o_format)) if o_format is not None else None
        do_format = tex.NVTE_QKV_Format(int(do_format)) if do_format is not None else None
        qkv_scale_inv_format = (
            tex.NVTE_QKV_Format(int(qkv_scale_inv_format))
            if qkv_scale_inv_format is not None
            else None
        )
        do_scale_inv_format = (
            tex.NVTE_QKV_Format(int(do_scale_inv_format))
            if do_scale_inv_format is not None
            else None
        )

        return tex.fused_attn_bwd(
            max_seqlen_q,
            max_seqlen_kv,
            attn_scale,
            p_dropout,
            set_zero,
            qkv_layout,
            o_format,
            do_format,
            dqkv_layout,
            qkv_scale_inv_format,
            do_scale_inv_format,
            bias_type,
            attn_mask_type,
            softmax_type,
            window_size,
            bottom_right_diagonal,
            deterministic,
            cu_seqlens_q,
            cu_seqlens_kv,
            Q,
            K,
            V,
            O,
            dO,
            fake_dtype,
            Aux_CTX_Tensors,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            s_quantizer,
            dp_quantizer,
            dqkv_quantizer,
            cuda_graph,
        )

    def copy_to_kv_cache(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cu_new_lens: torch.Tensor,
        cu_cached_lens: torch.Tensor,
        kv_format: NVTE_QKV_Format,
        b: int,
        max_ctx_len: int,
        max_seq_len: int,
        max_pages_per_seq: int,
        is_non_paged: bool,
    ) -> None:
        tex = self._get_tex()
        kv_format = tex.NVTE_QKV_Format(int(kv_format)) if kv_format is not None else None
        return tex.copy_to_kv_cache(
            new_k,
            new_v,
            k_cache,
            v_cache,
            page_table,
            cu_new_lens,
            cu_cached_lens,
            kv_format,
            b,
            max_ctx_len,
            max_seq_len,
            max_pages_per_seq,
            is_non_paged,
        )

    def convert_thd_to_bshd(
        self,
        tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
        b: int,
        max_seq_len: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.convert_thd_to_bshd(tensor, cu_seqlens, b, max_seq_len)

    def convert_bshd_to_thd(
        self,
        tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.convert_bshd_to_thd(tensor, cu_seqlens, t)

    def fused_rope_forward(
        self,
        input: torch.Tensor,
        freqs: torch.Tensor,
        start_positions: Optional[torch.Tensor],
        qkv_format: NVTE_QKV_Format,
        interleaved: bool,
        cu_seqlens: Optional[torch.Tensor],
        cp_size: int,
        cp_rank: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.fused_rope_forward(
            input,
            freqs,
            start_positions,
            qkv_format,
            interleaved,
            cu_seqlens,
            cp_size,
            cp_rank,
        )

    def fused_rope_backward(
        self,
        output_grads: torch.Tensor,
        freqs: torch.Tensor,
        start_positions: Optional[torch.Tensor],
        qkv_format: NVTE_QKV_Format,
        interleaved: bool,
        cu_seqlens: Optional[torch.Tensor],
        cp_size: int,
        cp_rank: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.fused_rope_backward(
            output_grads,
            freqs,
            start_positions,
            qkv_format,
            interleaved,
            cu_seqlens,
            cp_size,
            cp_rank,
        )

    def fused_qkv_rope_forward(
        self,
        qkv_input: torch.Tensor,
        q_freqs: torch.Tensor,
        k_freqs: torch.Tensor,
        start_positions: Optional[torch.Tensor],
        qkv_split_arg_list: List[int],
        qkv_format: NVTE_QKV_Format,
        interleaved: bool,
        cp_size: int,
        cp_rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.fused_qkv_rope_forward(
            qkv_input,
            q_freqs,
            k_freqs,
            start_positions,
            qkv_split_arg_list,
            qkv_format,
            interleaved,
            cp_size,
            cp_rank,
        )

    def fused_qkv_rope_backward(
        self,
        q_grad_out: torch.Tensor,
        k_grad_out: torch.Tensor,
        v_grad_out: torch.Tensor,
        q_freqs: torch.Tensor,
        k_freqs: torch.Tensor,
        qkv_split_arg_list: List[int],
        qkv_format: NVTE_QKV_Format,
        interleaved: bool,
        cp_size: int,
        cp_rank: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        qkv_format = tex.NVTE_QKV_Format(int(qkv_format)) if qkv_format is not None else None
        return tex.fused_qkv_rope_backward(
            q_grad_out,
            k_grad_out,
            v_grad_out,
            q_freqs,
            k_freqs,
            qkv_split_arg_list,
            qkv_format,
            interleaved,
            cp_size,
            cp_rank,
        )

    def fused_topk_with_score_function_fwd(
        self,
        logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        num_groups: Optional[int],
        group_topk: Optional[int],
        scaling_factor: Optional[float],
        score_function: str,
        expert_bias: Optional[torch.Tensor],
        routing_map_format: int = NVTERoutingMapFormat.BYTEMAP,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.fused_topk_with_score_function_fwd(
            logits,
            topk,
            use_pre_softmax,
            num_groups,
            group_topk,
            scaling_factor,
            score_function,
            expert_bias,
            int(routing_map_format),
            topk_indices,
        )

    def fused_topk_with_score_function_qb_fwd(
        self,
        logits: torch.Tensor,
        topk: int,
        scaling_factor: Optional[float],
        expert_bias: Optional[torch.Tensor],
        routing_map_format: int,
        topk_indices: Optional[torch.Tensor],
        histogram: Optional[torch.Tensor],
        bin_bounds: Optional[torch.Tensor],
        histogram_mode: int,
        bin_bounds_validated: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        tex = self._get_tex()
        return tex.fused_topk_with_score_function_qb_fwd(
            logits,
            topk,
            scaling_factor,
            expert_bias,
            int(routing_map_format),
            topk_indices,
            histogram,
            bin_bounds,
            int(histogram_mode),
            bin_bounds_validated,
        )

    def fused_topk_with_score_function_bwd(
        self,
        routing_map: torch.Tensor,
        intermediate_output: torch.Tensor,
        grad_probs: torch.Tensor,
        grad_logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        scaling_factor: Optional[float],
        score_function: str,
        use_dense_indices: bool = False,
        routing_map_format: int = NVTERoutingMapFormat.BYTEMAP,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fused_topk_with_score_function_bwd(
            routing_map,
            intermediate_output,
            grad_probs,
            grad_logits,
            topk,
            use_pre_softmax,
            scaling_factor,
            score_function,
            use_dense_indices,
            int(routing_map_format),
        )

    def fused_score_for_moe_aux_loss_fwd(
        self,
        logits: torch.Tensor,
        topk: int,
        score_function: str,
        routing_map_format: int = NVTERoutingMapFormat.BYTEMAP,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.fused_score_for_moe_aux_loss_fwd(
            logits,
            topk,
            score_function,
            int(routing_map_format),
        )

    def fused_score_for_moe_aux_loss_bwd(
        self,
        intermediate_output: torch.Tensor,
        grad_scores: torch.Tensor,
        grad_logits: torch.Tensor,
        topk: int,
        score_function: str,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fused_score_for_moe_aux_loss_bwd(
            intermediate_output,
            grad_scores,
            grad_logits,
            topk,
            score_function,
        )

    def fused_moe_aux_loss_fwd(
        self,
        probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        total_num_tokens: int,
        num_experts: int,
        num_rows: int,
        num_cols: int,
        topk: int,
        coeff: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.fused_moe_aux_loss_fwd(
            probs,
            tokens_per_expert,
            total_num_tokens,
            num_experts,
            num_rows,
            num_cols,
            topk,
            coeff,
        )

    def fused_moe_aux_loss_bwd(
        self,
        Const_buf: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        num_rows: int,
        num_cols: int,
        grad_aux_loss: torch.Tensor,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.fused_moe_aux_loss_bwd(
            Const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss
        )

    def dropout_fwd(
        self,
        input: torch.Tensor,
        dropout_probability: float,
        out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.dropout_fwd(input, dropout_probability, out)

    def dropout_bwd(
        self,
        grad_output: torch.Tensor,
        mask: torch.Tensor,
        dropout_probability: float,
        grad_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.dropout_bwd(grad_output, mask, dropout_probability, grad_input)

    def get_cublasLt_version(self) -> int:
        tex = self._get_tex()
        return tex.get_cublasLt_version()

    def get_cudnn_version(self) -> int:
        tex = self._get_tex()
        return tex.get_cudnn_version()

    def get_num_cublas_streams(self) -> int:
        tex = self._get_tex()
        return tex.get_num_cublas_streams()

    def thd_read_half_tensor(
        self,
        tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
        half_idx: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.thd_read_half_tensor(tensor, cu_seqlens, half_idx)

    def thd_second_half_lse_correction(
        self,
        lse: torch.Tensor,
        lse_per_step: torch.Tensor,
        cu_seqlens: torch.Tensor,
        lse_packed: bool,
    ) -> None:
        tex = self._get_tex()
        return tex.thd_second_half_lse_correction(lse, lse_per_step, cu_seqlens, lse_packed)

    def thd_read_second_half_lse(
        self,
        lse: torch.Tensor,
        cu_seqlens: torch.Tensor,
        lse_packed: bool,
        second_half_lse_seqlen: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.thd_read_second_half_lse(lse, cu_seqlens, lse_packed, second_half_lse_seqlen)

    def thd_out_correction(
        self,
        out: torch.Tensor,
        out_per_step: torch.Tensor,
        old_lse: torch.Tensor,
        lse: torch.Tensor,
        lse_per_step: torch.Tensor,
        cu_seqlens: torch.Tensor,
        only_second_half: bool,
        lse_packed: bool,
    ) -> None:
        tex = self._get_tex()
        return tex.thd_out_correction(
            out,
            out_per_step,
            old_lse,
            lse,
            lse_per_step,
            cu_seqlens,
            only_second_half,
            lse_packed,
        )

    def thd_grad_correction(
        self,
        grad: torch.Tensor,
        grad_per_step: torch.Tensor,
        cu_seqlens: torch.Tensor,
        first_half: str,
        second_half: str,
    ) -> None:
        tex = self._get_tex()
        return tex.thd_grad_correction(grad, grad_per_step, cu_seqlens, first_half, second_half)

    def thd_get_partitioned_indices(
        self,
        cu_seqlens: torch.Tensor,
        total_tokens: int,
        world_size: int,
        rank: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.thd_get_partitioned_indices(cu_seqlens, total_tokens, world_size, rank)

    def init_nvshmem_backend(
        self,
        process_group: Any,
    ) -> None:
        tex = self._get_tex()
        return tex.init_nvshmem_backend(process_group)

    def create_nvshmem_tensor(
        self,
        shape: List[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.create_nvshmem_tensor(shape, dtype)

    def nvshmem_send_on_current_stream(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        peer: int,
        signal: torch.Tensor,
    ) -> None:
        tex = self._get_tex()
        return tex.nvshmem_send_on_current_stream(src, dst, peer, signal)

    def nvshmem_wait_on_current_stream(
        self,
        signal: torch.Tensor,
        wait_kind: str,
    ) -> None:
        tex = self._get_tex()
        return tex.nvshmem_wait_on_current_stream(signal, wait_kind)

    def nvshmem_finalize(self) -> None:
        tex = self._get_tex()
        return tex.nvshmem_finalize()

    def multi_tensor_scale(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: float,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale)

    def multi_tensor_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        per_tensor: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.multi_tensor_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor)

    def multi_tensor_unscale_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        inv_scale: torch.Tensor,
        per_tensor: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.multi_tensor_unscale_l2norm(
            chunk_size, noop_flag, tensor_lists, inv_scale, per_tensor
        )

    def multi_tensor_adam(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        lr: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        step: int,
        mode: int,
        bias_correction: int,
        weight_decay: float,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_adam(
            chunk_size,
            noop_flag,
            tensor_lists,
            lr,
            beta1,
            beta2,
            epsilon,
            step,
            mode,
            bias_correction,
            weight_decay,
        )

    def multi_tensor_adam_param_remainder(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        lr: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        step: int,
        mode: int,
        bias_correction: int,
        weight_decay: float,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_adam_param_remainder(
            chunk_size,
            noop_flag,
            tensor_lists,
            lr,
            beta1,
            beta2,
            epsilon,
            step,
            mode,
            bias_correction,
            weight_decay,
        )

    def multi_tensor_adam_fp8(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        lr: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        step: int,
        mode: int,
        bias_correction: int,
        weight_decay: float,
        fp8_dtype: DType,
    ) -> None:
        tex = self._get_tex()
        fp8_dtype = tex.DType(int(fp8_dtype)) if fp8_dtype is not None else None
        return tex.multi_tensor_adam_fp8(
            chunk_size,
            noop_flag,
            tensor_lists,
            lr,
            beta1,
            beta2,
            epsilon,
            step,
            mode,
            bias_correction,
            weight_decay,
            fp8_dtype,
        )

    def multi_tensor_adam_capturable(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        lr: torch.Tensor,
        beta1: float,
        beta2: float,
        epsilon: float,
        step: torch.Tensor,
        mode: int,
        bias_correction: int,
        weight_decay: float,
        inv_scale: torch.Tensor,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_adam_capturable(
            chunk_size,
            noop_flag,
            tensor_lists,
            lr,
            beta1,
            beta2,
            epsilon,
            step,
            mode,
            bias_correction,
            weight_decay,
            inv_scale,
        )

    def multi_tensor_adam_capturable_master(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        lr: torch.Tensor,
        beta1: float,
        beta2: float,
        epsilon: float,
        step: torch.Tensor,
        mode: int,
        bias_correction: int,
        weight_decay: float,
        inv_scale: torch.Tensor,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_adam_capturable_master(
            chunk_size,
            noop_flag,
            tensor_lists,
            lr,
            beta1,
            beta2,
            epsilon,
            step,
            mode,
            bias_correction,
            weight_decay,
            inv_scale,
        )

    def multi_tensor_sgd(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        wd: float,
        momentum: float,
        dampening: float,
        lr: float,
        nesterov: bool,
        first_run: bool,
        wd_after_momentum: bool,
        scale: float,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_sgd(
            chunk_size,
            noop_flag,
            tensor_lists,
            wd,
            momentum,
            dampening,
            lr,
            nesterov,
            first_run,
            wd_after_momentum,
            scale,
        )

    def multi_tensor_compute_scale_and_scale_inv(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        max_fp8: float,
        force_pow_2_scales: bool,
        epsilon: float,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_compute_scale_and_scale_inv(
            chunk_size, noop_flag, tensor_lists, max_fp8, force_pow_2_scales, epsilon
        )

    def multi_tensor_compute_scale_inv_e8m0(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        block_len: int,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_compute_scale_inv_e8m0(
            chunk_size, noop_flag, tensor_lists, block_len
        )

    def bulk_overlap_ag_with_external_gemm(
        self,
        allgather_communicator: CommOverlap,
        send_stream: Any,
        recv_stream: Any,
    ) -> Any:
        tex = self._get_tex()
        return tex.bulk_overlap_ag_with_external_gemm(
            allgather_communicator, send_stream, recv_stream
        )

    def create_fp8_tensor_meta(self) -> FP8TensorMeta:
        tex = self._get_tex()
        return tex.FP8TensorMeta()

    def create_comm_overlap_helper(
        self,
        world_group: Optional[Any] = None,
        intra_node_group: Optional[Any] = None,
    ) -> "CommOverlapHelper":
        tex = self._get_tex()
        return tex.CommOverlapHelper(world_group, intra_node_group)

    def bulk_allocate(
        self,
        shapes: List[List[int]],
        dtypes: List[torch.dtype],
        device: Optional[torch.device] = None,
        alignments: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        tex = self._get_tex()
        return tex.bulk_allocate(shapes, dtypes, device, alignments)

    def copy_data_ptrs_to_device(
        self,
        tensors: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.copy_data_ptrs_to_device(tensors, device)

    def cusolvermp_ctx_create(
        self,
        nccl_comm_ptr: int,
        nranks: int,
        rank: int,
    ) -> int:
        tex = self._get_tex()
        return tex.cusolvermp_ctx_create(nccl_comm_ptr, nranks, rank)

    def cusolvermp_ctx_destroy(self, ctx_ptr: int) -> None:
        tex = self._get_tex()
        return tex.cusolvermp_ctx_destroy(ctx_ptr)

    def device_supports_multicast(self, device_id: int = -1) -> bool:
        tex = self._get_tex()
        return tex.device_supports_multicast(device_id)

    def fused_moe_aux_loss_fwd_graph_safe(
        self,
        probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        total_num_tokens: torch.Tensor,
        num_experts: int,
        num_rows: int,
        num_cols: int,
        topk: int,
        coeff: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.fused_moe_aux_loss_fwd_graph_safe(
            probs, tokens_per_expert, total_num_tokens,
            num_experts, num_rows, num_cols, topk, coeff,
        )

    def get_grouped_gemm_setup_workspace_size(self, num_tensors: int) -> int:
        tex = self._get_tex()
        return tex.get_grouped_gemm_setup_workspace_size(num_tensors)

    def get_stream_priority_range(self, device_id: int = -1) -> Tuple[int, int]:
        tex = self._get_tex()
        return tex.get_stream_priority_range(device_id)

    def multi_tensor_pad_last_dim(
        self,
        inputs: List[torch.Tensor],
        alignment: int,
    ) -> List[torch.Tensor]:
        tex = self._get_tex()
        return tex.multi_tensor_pad_last_dim(inputs, alignment)

    def multi_tensor_swizzle_scales_for_gemm_(
        self,
        tensors: List[Any],
        rowwise_usage: bool,
        columnwise_usage: bool,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_swizzle_scales_for_gemm_(tensors, rowwise_usage, columnwise_usage)

    def multi_tensor_swizzle_scales_for_gemm_unchecked_(
        self,
        tensors: List[Any],
        rowwise_usage: bool,
        columnwise_usage: bool,
    ) -> None:
        tex = self._get_tex()
        return tex.multi_tensor_swizzle_scales_for_gemm_unchecked_(tensors, rowwise_usage, columnwise_usage)

    def multi_tensor_transpose_to_bhsd(
        self,
        inputs: List[Optional[torch.Tensor]],
        original_format: str,
        outputs: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> List[Optional[torch.Tensor]]:
        tex = self._get_tex()
        if outputs is None:
            outputs = []
        return tex.multi_tensor_transpose_to_bhsd(inputs, original_format, outputs)

    def newton_schulz(
        self,
        ctx_ptr: int,
        m: int,
        n: int,
        x: torch.Tensor,
        num_iterations: int,
        coefficients: List[float],
    ) -> None:
        tex = self._get_tex()
        return tex.newton_schulz(ctx_ptr, m, n, x, num_iterations, coefficients)

    def nvte_built_with_cublasmp(self) -> bool:
        tex = self._get_tex()
        return tex.nvte_built_with_cublasmp()

    def scaled_swiglu(
        self,
        input: torch.Tensor,
        act_scales: torch.Tensor,
        quantizer: Any,
        glu_interleave_size: int = 0,
    ) -> Any:
        tex = self._get_tex()
        return tex.scaled_swiglu(input, act_scales, quantizer, glu_interleave_size)

    def scaled_dswiglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        act_scales: torch.Tensor,
        quantizer: Any,
        glu_interleave_size: int = 0,
        compute_scale_grad: bool = True,
    ) -> Tuple:
        tex = self._get_tex()
        return tex.scaled_dswiglu(
            grad, fwd_input, act_scales, quantizer, glu_interleave_size, compute_scale_grad,
        )

    def scaled_srelu(
        self,
        input: torch.Tensor,
        act_scales: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        tex = self._get_tex()
        return tex.scaled_srelu(input, act_scales, quantizer)

    def scaled_dsrelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        act_scales: torch.Tensor,
        quantizer: Any,
        compute_scale_grad: bool = True,
    ) -> Tuple:
        tex = self._get_tex()
        return tex.scaled_dsrelu(grad, fwd_input, act_scales, quantizer, compute_scale_grad)

    def scaled_clamped_swiglu(
        self,
        input: torch.Tensor,
        act_scales: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.7020000219345093,
        glu_linear_offset: float = 1.0,
        glu_interleave_size: int = 0,
    ) -> Any:
        tex = self._get_tex()
        return tex.scaled_clamped_swiglu(
            input, act_scales, quantizer, limit, alpha, glu_linear_offset, glu_interleave_size,
        )

    def scaled_clamped_dswiglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        act_scales: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.7020000219345093,
        glu_linear_offset: float = 1.0,
        glu_interleave_size: int = 0,
        compute_scale_grad: bool = True,
    ) -> Tuple:
        tex = self._get_tex()
        return tex.scaled_clamped_dswiglu(
            grad, fwd_input, act_scales, quantizer,
            limit, alpha, glu_linear_offset, glu_interleave_size, compute_scale_grad,
        )

    def splits_to_offsets_multi(
        self,
        split_sizes: torch.Tensor,
        device: torch.device,
        *,
        strides: List[int],
        include_leading_zero: List[bool],
        dtypes: List[torch.dtype],
        bulk_allocate: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        tex = self._get_tex()
        return tex.splits_to_offsets_multi(
            split_sizes, device,
            strides=strides,
            include_leading_zero=include_leading_zero,
            dtypes=dtypes,
            bulk_allocate=bulk_allocate,
        )

    def thd_copy_valid_tokens_from_per_split_to_rank_local(
        self,
        out: torch.Tensor,
        inp: torch.Tensor,
        cu_seqlens_padded: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> None:
        tex = self._get_tex()
        return tex.thd_copy_valid_tokens_from_per_split_to_rank_local(
            out, inp, cu_seqlens_padded, cu_seqlens,
        )

    def thd_cp_rank_order_to_sequence_order(
        self,
        inp: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cp_size: int,
        total_tokens: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.thd_cp_rank_order_to_sequence_order(inp, cu_seqlens, cp_size, total_tokens)

    def thd_sequence_order_to_cp_rank_order(
        self,
        inp: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cp_size: int,
        total_tokens: int,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.thd_sequence_order_to_cp_rank_order(inp, cu_seqlens, cp_size, total_tokens)

    def ubuf_built_with_mpi(self) -> bool:
        tex = self._get_tex()
        return tex.ubuf_built_with_mpi()

    def swizzle_scales_and_pack_ptrs_for_discrete_weights(
        self,
        data_tensors: List[torch.Tensor],
        scale_tensors: List[torch.Tensor],
        swizzle_type: str,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.grouped_mlp_experimental.swizzle_scales_and_pack_ptrs_for_discrete_weights(
            data_tensors, scale_tensors, swizzle_type, device,
        )

    def scaled_situglu(
        self,
        input: torch.Tensor,
        act_scales: torch.Tensor,
        quantizer: Any,
        beta1: float = 4.0,
        beta2: float = 25.0,
        glu_interleave_size: int = 0,
    ) -> Any:
        tex = self._get_tex()
        return tex.scaled_situglu(
            input, act_scales, quantizer, beta1, beta2, glu_interleave_size
        )

    def scaled_dsituglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        act_scales: torch.Tensor,
        quantizer: Any,
        beta1: float = 4.0,
        beta2: float = 25.0,
        glu_interleave_size: int = 0,
        compute_scale_grad: bool = True,
    ) -> Any:
        tex = self._get_tex()
        return tex.scaled_dsituglu(
            grad,
            fwd_input,
            act_scales,
            quantizer,
            beta1,
            beta2,
            glu_interleave_size,
            compute_scale_grad,
        )

    def group_scaled_swiglu(
        self,
        input_2h: torch.Tensor,
        prob: torch.Tensor,
        quantizer: Any,
        num_tensors: int,
        first_dims: Optional[List[int]] = None,
        last_dims: Optional[List[int]] = None,
        tensor_offsets: Optional[torch.Tensor] = None,
    ) -> Any:
        tex = self._get_tex()
        self._normalize_quantizer_dtype(quantizer)
        return tex.group_scaled_swiglu(
            input_2h, prob, quantizer, num_tensors, first_dims, last_dims, tensor_offsets
        )

    def group_scaled_clamped_swiglu(
        self,
        input_2h: torch.Tensor,
        prob: torch.Tensor,
        quantizer: Any,
        num_tensors: int,
        limit: float,
        alpha: float = 1.702,
        glu_linear_offset: float = 1.0,
        first_dims: Optional[List[int]] = None,
        last_dims: Optional[List[int]] = None,
        tensor_offsets: Optional[torch.Tensor] = None,
    ) -> Any:
        tex = self._get_tex()
        self._normalize_quantizer_dtype(quantizer)
        return tex.group_scaled_clamped_swiglu(
            input_2h,
            prob,
            quantizer,
            num_tensors,
            limit,
            alpha,
            glu_linear_offset,
            first_dims,
            last_dims,
            tensor_offsets,
        )

    def fused_mla_kv_rope_mxfp8(
        self,
        kv: torch.Tensor,
        k_rope: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        v_head_dim: int,
        key_quantizer: Any,
        value_quantizer: Any,
    ) -> List[Any]:
        tex = self._get_tex()
        self._normalize_quantizer_dtype(key_quantizer)
        self._normalize_quantizer_dtype(value_quantizer)
        return tex.fused_mla_kv_rope_mxfp8(
            kv, k_rope, cos, sin, v_head_dim, key_quantizer, value_quantizer
        )

    def get_cublasLt_build_version(self) -> int:
        tex = self._get_tex()
        return tex.get_cublasLt_build_version()

    def nvte_cublasmp_version(self) -> int:
        tex = self._get_tex()
        return tex.nvte_cublasmp_version()

    def get_nccl_comm_ptr(self, process_group: Any) -> int:
        tex = self._get_tex()
        return tex.get_nccl_comm_ptr(process_group)

    def ep_initialize(
        self,
        comm_ptr: int,
        group_name: str,
        num_experts: int,
        max_tokens_per_rank: int,
        max_recv_tokens_per_rank: int,
        hidden_dim: int,
        max_num_sms: int = 0,
        max_token_dtype: Any = None,
        zero_copy: bool = False,
        num_topk: int = 0,
        drop_on_overflow: bool = False,
    ) -> None:
        return self._tex_fn("ep_initialize")(
            comm_ptr,
            group_name,
            num_experts,
            max_tokens_per_rank,
            max_recv_tokens_per_rank,
            hidden_dim,
            max_num_sms,
            max_token_dtype,
            zero_copy,
            num_topk,
            drop_on_overflow,
        )

    def ep_finalize(self) -> None:
        return self._tex_fn("ep_finalize")()

    def ep_handle_mem_size(
        self,
        top_k: int,
        dispatch_output_per_expert_alignment: int = 0,
    ) -> int:
        return self._tex_fn("ep_handle_mem_size")(
            top_k, dispatch_output_per_expert_alignment
        )

    def ep_prepare(
        self,
        handle_mem: torch.Tensor,
        topk_idx: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        top_k: int,
        dispatch_output_per_expert_alignment: int,
        total_recv_tokens: torch.Tensor,
    ) -> Any:
        return self._tex_fn("ep_prepare")(
            handle_mem,
            topk_idx,
            tokens_per_expert,
            top_k,
            dispatch_output_per_expert_alignment,
            total_recv_tokens,
        )

    def ep_dispatch(
        self,
        handle_mem: torch.Tensor,
        topk_idx: torch.Tensor,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        recv_tokens: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        tokens_scale_inv: Optional[torch.Tensor] = None,
        recv_scale_inv: Optional[torch.Tensor] = None,
    ) -> Any:
        return self._tex_fn("ep_dispatch")(
            handle_mem,
            topk_idx,
            tokens,
            topk_weights,
            recv_tokens,
            recv_topk_weights,
            tokens_scale_inv,
            recv_scale_inv,
        )

    def ep_prepare_and_dispatch(
        self,
        handle_mem: torch.Tensor,
        topk_idx: torch.Tensor,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        total_recv_tokens: torch.Tensor,
        top_k: int,
        dispatch_output_per_expert_alignment: int,
        recv_tokens: Optional[torch.Tensor] = None,
        recv_topk_weights: Optional[torch.Tensor] = None,
        recv_scale_inv: Optional[torch.Tensor] = None,
        tokens_scale_inv: Optional[torch.Tensor] = None,
    ) -> Any:
        return self._tex_fn("ep_prepare_and_dispatch")(
            handle_mem,
            topk_idx,
            tokens,
            topk_weights,
            tokens_per_expert,
            total_recv_tokens,
            top_k,
            dispatch_output_per_expert_alignment,
            recv_tokens,
            recv_topk_weights,
            recv_scale_inv,
            tokens_scale_inv,
        )

    def ep_dispatch_bwd(
        self,
        handle_mem: torch.Tensor,
        grad: torch.Tensor,
        g_recv_topk_weights: torch.Tensor,
        grad_tokens: torch.Tensor,
        grad_topk_weights: torch.Tensor,
    ) -> Any:
        return self._tex_fn("ep_dispatch_bwd")(
            handle_mem, grad, g_recv_topk_weights, grad_tokens, grad_topk_weights
        )

    def ep_combine(
        self,
        handle_mem: torch.Tensor,
        expert_out: torch.Tensor,
        result: torch.Tensor,
    ) -> Any:
        return self._tex_fn("ep_combine")(handle_mem, expert_out, result)

    def ep_combine_bwd(
        self,
        handle_mem: torch.Tensor,
        grad: torch.Tensor,
        grad_expert_out: torch.Tensor,
        grad_scale_inv: Optional[torch.Tensor] = None,
        grad_expert_out_scale_inv: Optional[torch.Tensor] = None,
    ) -> Any:
        return self._tex_fn("ep_combine_bwd")(
            handle_mem, grad, grad_expert_out, grad_scale_inv, grad_expert_out_scale_inv
        )

    def ep_get_zero_copy(self) -> bool:
        return self._tex_fn("ep_get_zero_copy")()

    def ep_zero_copy_supported(self) -> bool:
        return self._tex_fn("ep_zero_copy_supported")()
    ##### END GENERATED OPS #####
