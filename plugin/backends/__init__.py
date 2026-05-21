# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Backend implementations for te_plugin.

Available backends:
  - cuda: NVIDIA CUDA (wraps transformer_engine_torch_nv pybind)
  - iluvatar: Iluvatar CoreX
  - kunlunxin: Kunlun XPU
  - metax: MetaX GPU
  - musa: Moore Threads MUSA
  - enflame: Enflame GCU
  - hygon: Hygon DCU
  - reference: Pure PyTorch reference implementation
  - flagos: FlagOS/FlagGems backend
"""
