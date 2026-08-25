from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.mamba_state_adapter import MambaStateAdapter


def test_adapter_creates_graph_buffer_with_stable_capacity():
    state = SimpleNamespace(
        conv=[torch.full((1, 2, 3), 2.0)],
        temporal=torch.full((1, 2, 4), 3.0),
    )
    adapter = MambaStateAdapter.from_state(state, slab_size=2)
    graph_buffer = adapter.create_graph_buffer(max_batch_size=4)
    batch = graph_buffer.load([torch.ones(1, 2, 3)], torch.ones(1, 2, 4))
    assert tuple(graph_buffer.conv[0].shape) == (1, 4, 3)
    assert tuple(graph_buffer.temporal.shape) == (1, 4, 4)
    assert batch.kernel_indices.tolist() == [0, 1]


def test_adapter_imports_state_and_materializes_logical_slots():
    state = SimpleNamespace(
        conv=[torch.full((1, 2, 3), 2.0)],
        temporal=torch.full((1, 2, 4), 3.0),
    )
    adapter = MambaStateAdapter.from_state(state, slab_size=2)
    batch = adapter.materialize(torch.tensor([0, 1]))
    assert adapter.capacity == 2
    assert batch.conv[0].shape == (1, 2, 3)
    assert batch.temporal.shape == (1, 2, 4)
    assert batch.conv[0].flatten().tolist() == [2.0] * 6
    assert batch.kernel_indices.tolist() == [0, 1]


def test_adapter_scatter_writes_back_kernel_updates():
    state = SimpleNamespace(
        conv=[torch.zeros((1, 2, 3))],
        temporal=torch.zeros((1, 2, 4)),
    )
    adapter = MambaStateAdapter.from_state(state, slab_size=2)
    batch = adapter.materialize(torch.tensor([0, 1]))
    batch.conv[0].fill_(6)
    batch.temporal.fill_(7)
    batch.scatter(adapter)
    restored = adapter.materialize(torch.tensor([0, 1]))
    assert restored.conv[0].flatten().tolist() == [6.0] * 6
    assert restored.temporal.flatten().tolist() == [7.0] * 8
