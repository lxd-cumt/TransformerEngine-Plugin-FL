# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Built-in operator implementations registration."""

from __future__ import annotations

import logging

from .registry import OpRegistry

logger = logging.getLogger(__name__)


def register_builtins(registry: OpRegistry) -> None:
    """Register all built-in backend operator implementations."""

    # Register CUDA (VENDOR) implementations
    try:
        from .backends.cuda.register_ops import register_builtins as register_cuda

        register_cuda(registry)
    except Exception as e:
        logger.debug(f"CUDA backend not loaded: {e}")

    # Register Iluvatar (VENDOR) implementations
    try:
        from .backends.iluvatar.register_ops import register_builtins as register_iluvatar

        register_iluvatar(registry)
    except Exception as e:
        logger.debug(f"Iluvatar backend not loaded: {e}")

    # Register KunlunXin (VENDOR) implementations
    try:
        from .backends.kunlunxin.register_ops import register_builtins as register_kunlunxin

        register_kunlunxin(registry)
    except Exception as e:
        logger.debug(f"KunlunXin backend not loaded: {e}")

    # Register MetaX (VENDOR) implementations
    try:
        from .backends.metax.register_ops import register_builtins as register_metax

        register_metax(registry)
    except Exception as e:
        logger.debug(f"MetaX backend not loaded: {e}")

    # Register MUSA (VENDOR) implementations
    try:
        from .backends.musa.register_ops import register_builtins as register_musa

        register_musa(registry)
    except Exception as e:
        logger.debug(f"MUSA backend not loaded: {e}")

    # Register Enflame (VENDOR) implementations
    try:
        from .backends.enflame.register_ops import register_builtins as register_enflame

        register_enflame(registry)
    except Exception as e:
        logger.debug(f"Enflame backend not loaded: {e}")

    # Register Hygon (VENDOR) implementations
    try:
        from .backends.hygon.register_ops import register_builtins as register_hygon

        register_hygon(registry)
    except Exception as e:
        logger.debug(f"Hygon backend not loaded: {e}")

    # Register FlagOS implementations
    try:
        from .backends.flagos.register_ops import register_builtins as register_flagos

        register_flagos(registry)
    except Exception as e:
        logger.debug(f"FlagOS backend not loaded: {e}")

    # Register Reference (pure PyTorch) implementations
    try:
        from .backends.reference.register_ops import register_builtins as register_reference

        register_reference(registry)
    except Exception as e:
        logger.debug(f"Reference backend not loaded: {e}")

