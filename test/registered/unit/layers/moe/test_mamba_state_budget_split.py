"""Budget split between the two Mamba state classes.

CPU-only, no GPU allocation: exercises `KVCacheConfigurator`'s sizing arithmetic
through lightweight stand-ins. The property under test is that enabling the int8
checkpoint pool *redistributes* the mamba budget (active pool bounded by
concurrency, remainder to cached prefixes) rather than sizing the active pool for
the whole budget and allocating the checkpoint pool on top of it.
"""

import types

import pytest
import torch

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.mamba_checkpoint_pool import (
    SPLIT_STATE_SOURCE,
    _sized_by_kv_budget,
)

NUM_LAYERS = 4
CONV_SHAPE = (128, 3)
TEMPORAL_SHAPE = (8, 64, 64)  # (heads, d_v, d_k)
GB = 1 << 30


class _ServerArgs:
    """Records overrides the way ServerArgs.override does, without the dataclass."""

    def __init__(self, **kwargs):
        self.max_mamba_cache_size = None
        self.max_running_requests = None
        self.int8_mamba_ckpt_size = None
        self.enable_int8_mamba_checkpoint = False
        self.mamba_full_memory_ratio = 1.0
        self.__dict__.update(kwargs)
        self._resolved_overrides = []

    def override(self, source, **fields):
        self._resolved_overrides.append((source, dict(fields)))
        self.__dict__.update(fields)


def _cache_params():
    return types.SimpleNamespace(
        shape=types.SimpleNamespace(conv=[CONV_SHAPE], temporal=TEMPORAL_SHAPE),
        dtype=types.SimpleNamespace(conv=torch.bfloat16, temporal=torch.bfloat16),
        layers=list(range(NUM_LAYERS)),
        # Active slot cost across all layers, matching BaseLinearStateParams.
        mamba_cache_per_req=(
            (CONV_SHAPE[0] * CONV_SHAPE[1] + TEMPORAL_SHAPE[0] * TEMPORAL_SHAPE[1] * TEMPORAL_SHAPE[2])
            * 2
            * NUM_LAYERS
        ),
    )


def _configurator(server_args, mamba_ratio=1):
    return types.SimpleNamespace(
        server_args=server_args,
        ps=types.SimpleNamespace(attn_dp_size=1),
        mambaish_config=types.SimpleNamespace(mamba2_cache_params=_cache_params()),
        _calculate_mamba_ratio=lambda: mamba_ratio,
    )


def _split_slots(server_args, has_spec_dec=False, mamba_ratio=1):
    return KVCacheConfigurator._split_state_active_slots(
        _configurator(server_args, mamba_ratio), has_spec_dec
    )


def test_split_is_off_without_the_checkpoint_pool():
    """No checkpoint pool means the radix still caches into the active pool, so
    the active pool must keep its whole-budget sizing."""
    assert _split_slots(_ServerArgs(max_running_requests=8)) is None


def test_split_needs_an_explicit_concurrency_bound():
    """resolve_max_num_reqs derives concurrency FROM max_mamba_cache_size, so the
    split cannot invent it -- it must come from a knob the user set."""
    args = _ServerArgs(enable_int8_mamba_checkpoint=True)
    assert _split_slots(args) is None

    args = _ServerArgs(enable_int8_mamba_checkpoint=True, max_running_requests=8)
    assert _split_slots(args) == 8


def test_explicit_mamba_cache_size_wins_over_running_requests():
    args = _ServerArgs(
        enable_int8_mamba_checkpoint=True,
        max_mamba_cache_size=4,
        max_running_requests=64,
    )
    assert _split_slots(args) == 4


def test_active_slots_scale_with_slots_per_request():
    """The extra_buffer ping-pong makes a request hold more than one active slot;
    resolve_max_num_reqs divides the pool by that same ratio, so the split has to
    multiply by it or concurrency silently drops."""
    args = _ServerArgs(enable_int8_mamba_checkpoint=True, max_running_requests=8)
    assert _split_slots(args, mamba_ratio=3) == 24


def test_speculative_decoding_keeps_the_old_sizing():
    """Draft intermediate buffers are sized off the active pool; shrinking it
    silently would under-allocate them."""
    args = _ServerArgs(enable_int8_mamba_checkpoint=True, max_running_requests=8)
    assert _split_slots(args, has_spec_dec=True) is None


def test_split_bounds_active_pool_by_concurrency_and_funds_checkpoints():
    args = _ServerArgs(enable_int8_mamba_checkpoint=True, max_running_requests=8)
    cfg = _configurator(args)
    per_req = cfg.mambaish_config.mamba2_cache_params.mamba_cache_per_req

    remaining = KVCacheConfigurator._handle_split_mamba_state(cfg, 8.0, 8)

    # Active pool tracks concurrency, NOT the budget.
    assert args.max_mamba_cache_size == 8
    # The leftover budget bought checkpoint slots, and int8 makes them cheaper
    # than active slots, so there are more of them.
    assert args.int8_mamba_ckpt_size > 8
    # Both pools were charged against the caller's budget.
    active_gb = 9 * per_req / GB
    assert remaining < 8.0 - active_gb
    assert remaining > 0


def test_split_charges_the_checkpoint_pool_against_the_budget():
    """The whole point: the checkpoint pool is accounted, so the KV budget shrinks
    by both pools rather than only the active one."""
    args = _ServerArgs(enable_int8_mamba_checkpoint=True, max_running_requests=4)
    cfg = _configurator(args)
    remaining = KVCacheConfigurator._handle_split_mamba_state(cfg, 8.0, 4)

    mamba_budget = 8.0 * 0.5  # mamba_full_memory_ratio=1.0 -> half the budget
    consumed = 8.0 - remaining
    # Consumed the mamba half (within one slot's rounding), not more, not less.
    assert consumed == pytest.approx(mamba_budget, abs=0.05)


def test_split_fails_loudly_when_concurrency_eats_the_budget():
    args = _ServerArgs(enable_int8_mamba_checkpoint=True, max_running_requests=4096)
    with pytest.raises(RuntimeError, match="Not enough GPU memory to split"):
        KVCacheConfigurator._handle_split_mamba_state(_configurator(args), 1.0, 4096)


def test_budgeted_pool_is_distinguishable_from_headroom_allocation():
    """The pool logs different accounting depending on who sized it; that check
    reads the override provenance, so it must survive the real override call."""
    args = _ServerArgs(enable_int8_mamba_checkpoint=True, max_running_requests=8)
    assert _sized_by_kv_budget(args) is False

    KVCacheConfigurator._handle_split_mamba_state(_configurator(args), 8.0, 8)
    assert _sized_by_kv_budget(args) is True
    assert args._resolved_overrides[-1][0] == SPLIT_STATE_SOURCE
