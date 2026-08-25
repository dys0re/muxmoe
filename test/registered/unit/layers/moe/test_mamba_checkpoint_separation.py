"""Two-class state separation: active pool vs radix checkpoint pool.

CPU-only. The active `MambaPool` holds state for *running* requests and is what
kernels (and a captured CUDA graph) touch; the radix's cached prefixes live in a
separate `MambaCheckpointPool` (int8). These tests pin the contract between them:
the transfer works identically whether the active state is one contiguous
allocation or slab-backed, and caching never grows the active pool.
"""

import types

import pytest
import torch

from sglang.srt.mem_cache.mamba_checkpoint_pool import MambaCheckpointPool
from sglang.srt.mem_cache.memory_pool import MambaPool

NUM_LAYERS = 2
CONV_SHAPE = (3, 2)
TEMPORAL_SHAPE = (1, 2, 2)  # (heads, d_v, d_k)


def _cache_params():
    """Minimal stand-in for BaseLinearStateParams (only shape/dtype are read)."""
    return types.SimpleNamespace(
        shape=types.SimpleNamespace(
            conv=[CONV_SHAPE],
            temporal=TEMPORAL_SHAPE,
            conv_shard_groups=None,
            conv_slice_axis=0,
        ),
        dtype=types.SimpleNamespace(conv=torch.float32, temporal=torch.float32),
        is_kda=False,
    )


def _pool(size=64, slab_size=None):
    return MambaPool(
        size=size,
        spec_state_size=0,
        cache_params=_cache_params(),
        mamba_layer_ids=list(range(NUM_LAYERS)),
        device="cpu",
        mamba_state_slab_size=slab_size,
    )


def _ckpt_pool(num_slots=8):
    heads, d_v, d_k = TEMPORAL_SHAPE
    return MambaCheckpointPool(
        num_layers=NUM_LAYERS,
        num_slots=num_slots,
        num_heads=heads,
        head_v_dim=d_v,
        head_k_dim=d_k,
        conv_shapes=[CONV_SHAPE],
        conv_dtype=torch.float32,
        device="cpu",
        temporal_dtype=torch.float32,
    )


def _fill(pool, slot, seed):
    """Write a recognizable state into one active slot, in either mode."""
    torch.manual_seed(seed)
    state = pool.State(
        conv=[torch.randn(NUM_LAYERS, 1, *CONV_SHAPE)],
        temporal=torch.randn(NUM_LAYERS, 1, *TEMPORAL_SHAPE),
    )
    pool.scatter_active_state(torch.tensor([slot]), state)
    return state


@pytest.mark.parametrize("slab_size", [None, 4])
def test_checkpoint_round_trip_matches_in_both_storage_modes(slab_size):
    """store_from_active -> load_to_active restores the state, contiguous or slab."""
    pool = _pool(slab_size=slab_size)
    ckpt = _ckpt_pool()
    written = _fill(pool, slot=5, seed=0)

    ckpt.store_from_active(pool, torch.tensor([5]), torch.tensor([3]))
    # Clobber the source slot so a passing restore cannot be reading stale state.
    pool.scatter_active_state(
        torch.tensor([5]),
        pool.State(
            conv=[torch.zeros(NUM_LAYERS, 1, *CONV_SHAPE)],
            temporal=torch.zeros(NUM_LAYERS, 1, *TEMPORAL_SHAPE),
        ),
    )
    ckpt.load_to_active(pool, torch.tensor([3]), torch.tensor([9]))

    restored = pool.gather_active_state(torch.tensor([9]))
    # conv is copied verbatim; only the temporal state is quantized.
    torch.testing.assert_close(restored.conv[0], written.conv[0])
    torch.testing.assert_close(
        restored.temporal, written.temporal, atol=0.02, rtol=0.02
    )


def test_slab_and_contiguous_restore_agree_bitwise():
    """The two storage modes are interchangeable, not merely each self-consistent."""
    results = []
    for slab_size in (None, 4):
        pool = _pool(slab_size=slab_size)
        ckpt = _ckpt_pool()
        _fill(pool, slot=2, seed=7)
        ckpt.store_from_active(pool, torch.tensor([2]), torch.tensor([1]))
        ckpt.load_to_active(pool, torch.tensor([1]), torch.tensor([6]))
        results.append(pool.gather_active_state(torch.tensor([6])))

    torch.testing.assert_close(results[0].conv[0], results[1].conv[0])
    torch.testing.assert_close(results[0].temporal, results[1].temporal)


def test_caching_states_does_not_grow_the_active_slab_pool():
    """The point of the separation: cached prefixes cost checkpoint slots, not
    active ones. Occupancy must track concurrency, not the number of cached
    states."""
    pool = _pool(size=1024, slab_size=4)
    ckpt = _ckpt_pool(num_slots=32)

    # One "running request" caches 16 successive prefixes, as a long prefill
    # would. Every checkpoint leaves the active slot where it was.
    _fill(pool, slot=0, seed=1)
    for _ in range(16):
        # Slot lifecycle is the radix's job; mirror it so the accounting is real.
        ckpt.store_from_active(pool, torch.tensor([0]), ckpt.alloc(1))

    assert pool.mamba_state_store.live_slots == 1
    assert pool.mamba_state_store.capacity == 4  # still the initial slab
    assert ckpt.available_size() == 32 - 16


def test_active_state_dtypes_reports_storage_dtype_without_allocating():
    """load_to_active needs the destination precision before it dequantizes."""
    for slab_size in (None, 4):
        conv_dtypes, temporal_dtype = _pool(slab_size=slab_size).active_state_dtypes()
        assert temporal_dtype is torch.float32
        assert conv_dtypes == [torch.float32]
    # Querying dtypes must not have forced any slab rows into existence.
    assert _pool(slab_size=4).mamba_state_store.live_slots == 0
