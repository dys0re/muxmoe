import torch

from sglang.srt.mem_cache.mamba_state_view import MambaStateView


def test_state_view_materializes_kernel_compatible_batch():
    view = MambaStateView.from_shapes(
        conv_shapes=[(1, 3), (2, 3)],
        temporal_shape=(2, 2, 2),
        conv_dtype=torch.float32,
        temporal_dtype=torch.float32,
        capacity=1,
        slab_size=2,
        device="cpu",
    )
    batch = view.materialize([0, 3])
    assert [tuple(t.shape) for t in batch.conv] == [(1, 2, 3), (2, 2, 3)]
    assert tuple(batch.temporal.shape) == (2, 2, 2, 2)
    assert batch.physical_ids.tolist() == [0, 3]


def test_state_view_scatter_preserves_compact_update():
    view = MambaStateView.from_shapes(
        conv_shapes=[(1, 2)],
        temporal_shape=(1, 2),
        conv_dtype=torch.float32,
        temporal_dtype=torch.float32,
        capacity=1,
        slab_size=2,
        device="cpu",
    )
    batch = view.materialize([0, 2])
    batch.conv[0].fill_(4)
    batch.temporal.fill_(5)
    batch.scatter(view.store)
    restored = view.materialize([0, 2])
    assert restored.conv[0].flatten().tolist() == [4] * 4
    assert restored.temporal.flatten().tolist() == [5] * 4
