"""Slab-backed Mamba state runtime: pool construction and slot lifecycle.

CPU-only. Builds a real `MambaPool` in slab mode and checks that (a) the
contiguous state is never allocated, (b) occupancy tracks concurrently live
requests rather than the slot id range, and (c) state round-trips through the
materialize -> kernel -> scatter path the GDN backend uses.
"""

import types

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import MambaPool


def _cache_params(num_layers=2, conv_dim=3, conv_win=2, hv=1, k=2, v=2):
    """Minimal stand-in for BaseLinearStateParams (only shape/dtype are read)."""
    return types.SimpleNamespace(
        shape=types.SimpleNamespace(
            conv=[(conv_dim, conv_win)],
            temporal=(hv, v, k),
            conv_shard_groups=None,
            conv_slice_axis=0,
        ),
        dtype=types.SimpleNamespace(conv=torch.float32, temporal=torch.float32),
        is_kda=False,
    )


def _pool(size=64, slab_size=4, num_layers=2):
    return MambaPool(
        size=size,
        spec_state_size=0,
        cache_params=_cache_params(num_layers=num_layers),
        mamba_layer_ids=list(range(num_layers)),
        device="cpu",
        mamba_state_slab_size=slab_size,
    )


def test_slab_pool_does_not_allocate_contiguous_state():
    pool = _pool(size=1024, slab_size=4)
    assert pool.slab_state_enabled is True
    # The (num_layers, size + 1, ...) tensors must never be built.
    assert pool.mamba_cache is None
    # Only the initial slab is resident, not 1025 slots.
    assert pool.mamba_state_store.capacity == 4


def test_slab_pool_rejects_unsupported_combinations():
    for kwargs, expected in [
        ({"speculative_num_draft_tokens": 4}, "speculative"),
        ({"envelope_layout": True}, "envelope"),
        ({"enable_linear_replayssm": True}, "ReplaySSM"),
    ]:
        with pytest.raises(ValueError, match=expected):
            MambaPool(
                size=16,
                spec_state_size=4,
                cache_params=_cache_params(),
                mamba_layer_ids=[0, 1],
                device="cpu",
                mamba_state_slab_size=4,
                **kwargs,
            )


def test_slab_pool_guards_contiguous_only_consumers():
    pool = _pool()
    for call in (
        lambda: pool.mamba2_layer_cache(0),
        lambda: pool.get_contiguous_buf_infos(),
    ):
        with pytest.raises(NotImplementedError, match="slab"):
            call()


def test_slab_state_round_trips_through_materialize_and_scatter():
    pool = _pool(num_layers=2)
    physical = pool.slab_bind([7, 9])

    batch = pool.slab_materialize(physical)
    # Compact tensors are [num_layers, bs, ...] and start zeroed.
    assert batch.temporal.shape == (2, 2, 1, 2, 2)
    assert batch.conv[0].shape == (2, 2, 3, 2)
    assert batch.temporal.abs().sum().item() == 0

    # Simulate an in-place kernel update on the compact batch, then write back.
    batch.temporal[:, 0].fill_(4.0)
    batch.conv[0][:, 1].fill_(5.0)
    pool.slab_scatter(physical, batch)

    reread = pool.slab_materialize(pool.slab_bind([7, 9]))
    assert reread.temporal[:, 0].unique().tolist() == [4.0]
    assert reread.temporal[:, 1].unique().tolist() == [0.0]
    assert reread.conv[0][:, 1].unique().tolist() == [5.0]
    assert reread.conv[0][:, 0].unique().tolist() == [0.0]


def test_slab_occupancy_is_bounded_by_concurrent_requests():
    # Slot ids come from the whole 1..size range; occupancy must follow the
    # number of live requests, not the id range.
    pool = _pool(size=4096, slab_size=8)
    for logical in (3, 1500, 4000, 77):
        pool.slab_bind([logical])
        pool.slab_release([logical])
    assert pool.mamba_state_store.capacity == 8
    assert pool.mamba_state_store.live_slots == 0


def test_slab_slot_reuse_starts_from_clean_state():
    pool = _pool()
    first = pool.slab_bind([5])
    batch = pool.slab_materialize(first)
    batch.temporal.fill_(3.0)
    pool.slab_scatter(first, batch)

    pool.slab_release([5])
    # A different request takes the recycled row and must not see stale state.
    second = pool.slab_bind([6])
    assert second.tolist() == first.tolist()
    assert pool.slab_materialize(second).temporal.abs().sum().item() == 0


def test_slab_pool_defines_attrs_the_scheduler_reads():
    # _init_slab_state returns before the contiguous path's tail, so any attr
    # set down there must be replicated. Missing replayssm_write_pos crashed
    # the scheduler on the first alloc().
    pool = _pool()
    for attr in (
        "replayssm_write_pos",
        "replayssm_cache_base",
        "replayssm_is_flush",
        "replayssm_is_kda",
        "conv_shard_groups",
        "conv_slice_axis",
        "num_mamba_layers",
        "mem_usage",
    ):
        assert hasattr(pool, attr), f"slab pool missing {attr}"


def test_slab_padded_rows_use_reserved_zero_row():
    # CUDA-graph batches are padded to a bucket size with slot -1. Those rows
    # must resolve to the reserved zero row instead of consuming slab capacity.
    pool = _pool(slab_size=4)
    store = pool.mamba_state_store
    physical = store.bind_padded([5, -1, 7, -1])
    assert physical[1].item() == store.pad_row
    assert physical[3].item() == store.pad_row
    # Only the two real slots were allocated, and neither took row 0.
    assert store.live_slots == 2
    assert physical[0].item() != store.pad_row
    assert physical[2].item() != store.pad_row


def test_slab_padded_row_stays_zero_after_real_writes():
    pool = _pool()
    store = pool.mamba_state_store
    physical = store.bind_padded([2, -1])
    conv, temporal = pool.slab_layer_state(0, physical)
    temporal.fill_(5.0)
    pool.slab_layer_scatter(0, physical, conv, temporal)
    # Writing through a padded batch must not corrupt the shared zero row for
    # the next replay; re-materialize a fresh padded batch and check it.
    fresh = store.bind_padded([-1])
    _, pad_state = pool.slab_layer_state(0, fresh)
    assert pad_state.abs().sum().item() == 0


def test_slab_local_index_keeps_padding_sentinel():
    pool = _pool()
    idx = pool.slab_local_index([7, 2], [7, -1, 2], torch.device("cpu"), torch.int32)
    assert idx.tolist() == [0, -1, 1]


def test_slab_copy_from_duplicates_slot_state():
    # Radix COW copies a whole slot; on slabs this duplicates the physical rows.
    pool = _pool()
    src = pool.slab_bind([3])
    conv, temporal = pool.slab_layer_state(0, src)
    temporal.fill_(9.0)
    pool.slab_layer_scatter(0, src, conv, temporal)

    pool.copy_from(torch.tensor([3]), torch.tensor([4]))
    _, copied = pool.slab_layer_state(0, pool.slab_bind([4]))
    assert copied.unique().tolist() == [9.0]
    # Source is untouched and the two slots are distinct rows.
    assert pool.slab_bind([3]).tolist() != pool.slab_bind([4]).tolist()


def test_slab_local_index_maps_union_positions():
    # Track slots live alongside batch slots in one materialized tensor; the
    # kernels need each set's position inside that union.
    pool = _pool()
    union = [7, 2, 5]
    idx = pool.slab_local_index(union, [5, 7], torch.device("cpu"), torch.int32)
    assert idx.tolist() == [2, 0]


def test_slab_clear_slots_zeroes_logical_rows():
    # clear_slots is on the alloc path for a freshly assigned slot, so it must
    # work on slabs rather than reject.
    pool = _pool()
    physical = pool.slab_bind([6])
    conv, temporal = pool.slab_layer_state(0, physical)
    temporal.fill_(7.0)
    pool.slab_layer_scatter(0, physical, conv, temporal)

    pool.clear_slots(torch.tensor([6]))
    _, after = pool.slab_layer_state(0, pool.slab_bind([6]))
    assert after.abs().sum().item() == 0


def test_slab_layer_state_round_trips_per_layer():
    pool = _pool(num_layers=3)
    physical = pool.slab_bind([2, 8])

    conv, temporal = pool.slab_layer_state(1, physical)
    assert temporal.shape == (2, 1, 2, 2)  # [bs, HV, V, K]
    assert conv[0].shape == (2, 3, 2)  # [bs, dim, K-1]

    # Simulate an in-place kernel update on layer 1 only.
    temporal.fill_(2.0)
    conv[0].fill_(3.0)
    pool.slab_layer_scatter(1, physical, conv, temporal)

    _, layer1 = pool.slab_layer_state(1, physical)
    assert layer1.unique().tolist() == [2.0]
    # Neighbouring layers must be untouched.
    for other in (0, 2):
        _, state = pool.slab_layer_state(other, physical)
        assert state.abs().sum().item() == 0


def test_slab_state_persists_across_decode_steps():
    # The property that makes GDN decode correct: a kernel's in-place update to
    # the compact batch must be visible on the next step after scatter-back.
    pool = _pool(num_layers=2)
    slot = 11
    for step in range(1, 4):
        physical = pool.slab_bind([slot])
        conv, temporal = pool.slab_layer_state(0, physical)
        # Recurrent-style accumulate, as the real kernel does.
        temporal += step
        pool.slab_layer_scatter(0, physical, conv, temporal)

    _, final = pool.slab_layer_state(0, pool.slab_bind([slot]))
    assert final.unique().tolist() == [6.0]  # 1 + 2 + 3


def test_slab_decode_state_is_isolated_between_concurrent_slots():
    pool = _pool()
    physical = pool.slab_bind([4, 5])
    conv, temporal = pool.slab_layer_state(0, physical)
    temporal[0].fill_(1.0)
    temporal[1].fill_(2.0)
    pool.slab_layer_scatter(0, physical, conv, temporal)

    _, reread = pool.slab_layer_state(0, physical)
    assert reread[0].unique().tolist() == [1.0]
    assert reread[1].unique().tolist() == [2.0]


def test_slab_state_bytes_grows_only_with_slabs():
    pool = _pool(size=4096, slab_size=4)
    before = pool.slab_state_bytes()
    # Row 0 is the reserved padding row, so a 4-row slab holds 3 real slots.
    pool.slab_bind([10, 11, 12])
    assert pool.slab_state_bytes() == before
    # The next concurrent slot forces exactly one more slab.
    pool.slab_bind([99])
    assert pool.slab_state_bytes() == before * 2
