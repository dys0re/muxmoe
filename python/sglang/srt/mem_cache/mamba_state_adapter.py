from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.graph_state_buffer import GraphStateBuffer
from sglang.srt.mem_cache.mamba_state_store import MambaStateStore


@dataclass
class MambaStateAdapterBatch:
    """Kernel-facing compact state plus batch-local kernel indices."""

    conv: list[torch.Tensor]
    temporal: torch.Tensor
    physical_ids: torch.Tensor
    kernel_indices: torch.Tensor

    def scatter(self, adapter: "MambaStateAdapter") -> None:
        for layer, values in enumerate(self.conv):
            adapter.store.scatter(
                f"conv_{layer}", self.physical_ids, values
            )
        adapter.store.scatter("temporal", self.physical_ids, self.temporal)


class MambaStateAdapter:
    """Bridge an existing MambaPool.State to slab-backed batch access.

    This adapter is intentionally independent of MambaPool so it can be adopted
    by one backend at a time. The source state is imported once; subsequent
    materialize/scatter calls use the stable logical slot contract.
    """

    def __init__(self, store: MambaStateStore, conv_layers: int):
        self.store = store
        self.conv_layers = conv_layers

    @classmethod
    def from_state(cls, state, *, slab_size: int) -> "MambaStateAdapter":
        fields = {
            **{f"conv_{i}": tensor for i, tensor in enumerate(state.conv)},
            "temporal": state.temporal,
        }
        store = MambaStateStore.from_state_tensors(fields, slab_size=slab_size)
        store.import_fields(fields)
        return cls(store, len(state.conv))

    @property
    def capacity(self) -> int:
        return self.store.capacity

    def materialize(self, logical_ids: torch.Tensor) -> MambaStateAdapterBatch:
        physical_ids = self.store.ensure_logical_slots(logical_ids.tolist())
        conv = [
            self.store.field(f"conv_{layer}", physical_ids)
            for layer in range(self.conv_layers)
        ]
        temporal = self.store.field("temporal", physical_ids)
        return MambaStateAdapterBatch(
            conv=conv,
            temporal=temporal,
            physical_ids=physical_ids,
            kernel_indices=torch.arange(
                physical_ids.numel(), dtype=torch.long, device=physical_ids.device
            ),
        )

    def create_graph_buffer(self, *, max_batch_size: int) -> GraphStateBuffer:
        sample = self.materialize(torch.tensor([0], device=self.store.device))
        return GraphStateBuffer(
            conv_shapes=[tuple(t.shape[0:1] + t.shape[2:]) for t in sample.conv],
            temporal_shape=tuple(sample.temporal.shape[0:1] + sample.temporal.shape[2:]),
            max_batch_size=max_batch_size,
            conv_dtype=sample.conv[0].dtype,
            temporal_dtype=sample.temporal.dtype,
            device=self.store.device,
        )
