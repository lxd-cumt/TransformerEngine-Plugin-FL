# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""TE Plugin — operator dispatch infrastructure and backend implementations.

This package provides:
  - Plugin management: registry, policy-based dispatch, stub module
  - Backend implementations under transformer_engine_plugin_fl.backends.*

Architecture:
  sys.modules["transformer_engine_torch"] = stub (pure Python, enums + OpManager dispatch)
  CUDA Backend → import transformer_engine_torch_nv (original pybind .so) → register ops
  Chip-X Backend → custom implementation → register ops

Usage from TransformerEngine (only 3 lines added to TE):
    try:
        from transformer_engine_plugin_fl import load_plugins
        load_plugins()
    except ImportError:
        pass
"""

from transformer_engine_plugin_fl.types import BackendImplKind, OpImpl, match_token
from transformer_engine_plugin_fl.registry import OpRegistry
from transformer_engine_plugin_fl.policy import (
    SelectionPolicy,
    PolicyManager,
    get_policy,
    set_global_policy,
    reset_global_policy,
    policy_context,
    policy_from_env,
    get_policy_epoch,
    bump_policy_epoch,
    with_strict_mode,
    with_preference,
    with_allowed_vendors,
    with_denied_vendors,
    PREFER_DEFAULT,
    PREFER_VENDOR,
    PREFER_REFERENCE,
    VALID_PREFER_VALUES,
)
from transformer_engine_plugin_fl.manager import OpManager, get_default_manager, reset_default_manager
from transformer_engine_plugin_fl.discovery import (
    discover_plugin,
    discover_from_entry_points,
    discover_from_env_modules,
    get_discovered_plugin,
    clear_discovered_plugin,
    PLUGIN_GROUP,
    PLUGIN_MODULES_ENV,
)
from transformer_engine_plugin_fl.stub import install_stub


def load_plugins():
    """Entry point called from TransformerEngine.

    Triggers:
      1. Plugin discovery (entry_points + env modules)
      2. Backend registration (builtin_ops + discovered plugins)
      3. Install stub module as sys.modules["transformer_engine_torch"]
    """
    manager = get_default_manager()
    manager.ensure_initialized()
    install_stub(manager)
