from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from sglang.srt.mem_cache.mamba_state_store import MambaStateStore


@dataclass
class MambaStateBatch:
    """Compact per-layer state tensors for one forward batch."""

    conv: list[torch.Tensor]
    temporal: torch.Tensor
    physical_ids: torch.Tensor

    def scatter(self, store: MambaStateStore) -> None:
        for layer, tensor in enumerate(self.conv):
            store.scatter(f"conv_{layer}", self.physical_ids, tensor)
        store.scatter("temporal", self.physical_ids, self.temporal)


class MambaStateView:
    """Compatibility view between slab storage and existing Mamba backends.

    Existing kernels consume ``conv[layer]`` and ``temporal[layer]`` tensors. This
    adapter materializes those tensors for a batch and provides an explicit
    scatter-back operation; no kernel receives a slab list or a logical id.
    """

    def __init__(self, store: MambaStateStore, conv_layers: int) -> None:
        self.store = store
        self.conv_layers = conv_layers

    @classmethod
    def from_shapes(
        cls,
        *,
        conv_shapes: list[tuple[int, ...]],
        temporal_shape: tuple[int, ...],
        conv_dtype: torch.dtype,
        temporal_dtype: torch.dtype,
        capacity: int,
        slab_size: int,
        device: torch.device | str,
    ) -> "MambaStateView":
        fields = {f"conv_{i}": shape for i, shape in enumerate(conv_shapes)}
        fields["temporal"] = temporal_shape
        dtypes = {f"conv_{i}": conv_dtype for i in range(len(conv_shapes))}
        dtypes["temporal"] = temporal_dtype
        return cls(
            MambaStateStore(
                fields,
                capacity=capacity,
                slab_size=slab_size,
                dtype=dtypes,
                device=device,
            ),
            len(conv_shapes),
        )

    def ensure(self, logical_ids: Iterable[int]) -> torch.Tensor:
        return self.store.ensure_logical_slots(logical_ids)

    def materialize(self, logical_ids: Iterable[int]) -> MambaStateBatch:
        physical = self.ensure(logical_ids)
        conv = [
            self.store.field(f"conv_{layer}", physical)
            for layer in range(self.conv_layers)
        ]
        temporal = self.store.field("temporal", physical)
        return MambaStateBatch(conv=conv, temporal=temporal, physical_ids=physical)
