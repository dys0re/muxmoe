import pytest
import torch

from sglang.srt.mem_cache.mamba_state_store import MambaStateStore


def test_state_store_grows_without_changing_existing_logical_ids():
    store = MambaStateStore(
        {"temporal": (2, 4)}, capacity=1, slab_size=2,
        dtype={"temporal": torch.float32}, device="cpu"
    )
    first = store.bind([0])
    store.scatter("temporal", first, torch.full((2, 1, 4), 3.0))
    store.ensure_slots(5)
    second = store.bind([0, 4])
    # Capacity rounds up to a slab multiple: partial slabs would desync the
    # divmod(physical, slab_size) addressing from the real layout.
    assert store.capacity == 6
    assert store.slab_count == 3
    assert second.tolist() == [0, 4]
    assert store.field("temporal", first).flatten().tolist() == [3] * 8


def test_state_store_addresses_correct_row_after_partial_growth():
    # Regression: ensure_slots used to allocate a partial final slab, so after
    # growing in uneven steps divmod(physical, slab_size) pointed at the wrong
    # row (or past the end of a short slab) and reads/writes silently crossed
    # slots. Grow unevenly, then check every slot round-trips its own value.
    store = MambaStateStore(
        {"temporal": (1, 1)}, capacity=1, slab_size=2,
        dtype={"temporal": torch.float32}, device="cpu"
    )
    store.ensure_slots(5)
    ids = store.bind([0, 1, 2, 3, 4])
    values = torch.tensor([[[10.0], [11.0], [12.0], [13.0], [14.0]]])
    store.scatter("temporal", ids, values)
    for slot, expected in enumerate([10.0, 11.0, 12.0, 13.0, 14.0]):
        got = store.field("temporal", torch.tensor([slot])).flatten().tolist()
        assert got == [expected], f"slot {slot} read {got}, expected {expected}"


def test_state_store_clear_only_selected_slots():
    store = MambaStateStore(
        {"temporal": (1, 2)}, capacity=2, slab_size=2,
        dtype={"temporal": torch.float32}, device="cpu"
    )
    ids = store.bind([0, 1])
    store.scatter("temporal", ids, torch.full((1, 2, 2), 7.0))
    store.clear(ids[:1])
    assert store.field("temporal", ids).flatten().tolist() == [0, 0, 7, 7]




def _reuse_store(capacity=2, slab_size=2, max_capacity=None):
    return MambaStateStore(
        {"temporal": (1, 2)}, capacity=capacity, slab_size=slab_size,
        dtype={"temporal": torch.float32}, device="cpu",
        reuse_released_slots=True, max_capacity=max_capacity,
    )


def test_reuse_mode_keeps_capacity_flat_for_sparse_logical_ids():
    # The real MambaSlotAllocator hands out ids from the whole 1..size range, so
    # identity mapping would fault in a slab per distinct id and defeat fixed
    # occupancy. Reuse mode must keep capacity at the concurrent-slot count.
    store = _reuse_store(capacity=0, slab_size=4)
    for logical in (5, 900, 4242):
        ids = store.bind([logical])
        store.scatter("temporal", ids, torch.full((1, 1, 2), 1.0))
        store.release([logical])
    # One slab total, regardless of how far apart the logical ids are.
    assert store.slab_count == 1
    assert store.capacity == 4
    assert store.live_slots == 0
    assert store.available_logical_slots() == 1


def test_reuse_mode_recycles_slot_with_clean_state():
    store = _reuse_store()
    first = store.bind([11])
    store.scatter("temporal", first, torch.full((1, 1, 2), 9.0))
    store.release([11])

    # A different logical id takes over the freed physical slot, and must not
    # observe the previous request's state.
    second = store.bind([12])
    assert second.tolist() == first.tolist()
    assert store.field("temporal", second).flatten().tolist() == [0, 0]
    assert store.available_logical_slots() == 0


def test_reuse_mode_keeps_concurrent_slots_distinct():
    store = _reuse_store(capacity=0, slab_size=2)
    ids = store.bind([3, 77])
    assert len(set(ids.tolist())) == 2
    store.scatter("temporal", ids, torch.tensor([[[1.0, 1.0], [2.0, 2.0]]]))
    assert store.field("temporal", ids).flatten().tolist() == [1, 1, 2, 2]
    # Releasing one must not disturb the other's state.
    store.release([3])
    assert store.field("temporal", store.bind([77])).flatten().tolist() == [2, 2]


def test_reuse_mode_raises_when_concurrent_slots_exhausted():
    # max_capacity counts physical rows; row 0 is reserved for padding, so
    # max_capacity=3 leaves exactly two usable slots.
    store = _reuse_store(capacity=0, slab_size=1, max_capacity=3)
    store.bind([1, 2])
    with pytest.raises(RuntimeError, match="exhausted"):
        store.bind([3])
    # Freeing one makes room again without growing capacity.
    store.release([1])
    store.bind([3])
    assert store.capacity == 3


def test_identity_mode_is_unchanged_by_default():
    store = MambaStateStore(
        {"temporal": (1, 2)}, capacity=2, slab_size=2,
        dtype={"temporal": torch.float32}, device="cpu"
    )
    assert store.bind([0, 1]).tolist() == [0, 1]
    # Identity mapping still grows to cover the id, which the contiguous-pool
    # import helpers depend on.
    assert store.bind([5]).tolist() == [5]
    assert store.capacity == 6


def test_state_store_rebinding_reclaims_idle_slot_with_clean_state():
    # Regression: release() marked an id idle but bind() never reclaimed it, so
    # available_logical_slots() desynced (stayed >0 after the id was rebound).
    # Fixed-occupancy streaming needs the rebound id to reuse the same physical
    # slot with zeroed state and the idle count to return to 0.
    store = MambaStateStore(
        {"temporal": (1, 2)}, capacity=2, slab_size=2,
        dtype={"temporal": torch.float32}, device="cpu"
    )
    ids = store.bind([0, 1])
    store.scatter("temporal", ids, torch.full((1, 2, 2), 5.0))

    store.release([1])
    assert store.available_logical_slots() == 1

    # Rebinding id 1 (as the allocator would) reuses physical slot 1, keeps
    # capacity fixed, drains the idle count, and reads back zeros — not stale 5s.
    reused = store.bind([1])
    assert reused.tolist() == [1]
    assert store.capacity == 2
    assert store.available_logical_slots() == 0
    assert store.field("temporal", reused).flatten().tolist() == [0, 0]
    # The still-live logical id 0 keeps its original physical mapping and value.
    assert store.field("temporal", store.bind([0])).flatten().tolist() == [5, 5]


def test_state_store_releases_and_reuses_logical_slots():
    store = MambaStateStore(
        {"temporal": (1, 2)}, capacity=2, slab_size=2,
        dtype={"temporal": torch.float32}, device="cpu"
    )
    ids = store.bind([0, 1])
    store.scatter("temporal", ids, torch.ones(1, 2, 2))
    store.release([1])
    assert store.available_logical_slots() == 1
    assert store.field("temporal", torch.tensor([1])).flatten().tolist() == [0, 0]
    store = MambaStateStore(
        {"temporal": (1, 2)}, capacity=2, slab_size=2,
        dtype={"temporal": torch.float32}, device="cpu"
    )
    ids = store.bind([0, 1])
    store.scatter("temporal", ids[:1], torch.ones(1, 1, 2))
    store.copy_slots(ids[:1], ids[1:])
    assert len(list(store.iter_field_slabs("temporal"))) == 1
    assert store.state_fields() == ("temporal",)
    assert store.field("temporal", ids[1:]).flatten().tolist() == [1, 1]
