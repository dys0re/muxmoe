from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GraphStateBatch:
    conv: list[torch.Tensor]
    temporal: torch.Tensor
    kernel_indices: torch.Tensor


class GraphStateBuffer:
    """Fixed-address compact state buffers for one CUDA-graph batch bucket."""

    def __init__(
        self,
        *,
        conv_shapes: list[tuple[int, ...]],
        temporal_shape: tuple[int, ...],
        max_batch_size: int,
        conv_dtype: torch.dtype,
        temporal_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.device = torch.device(device)
        self.max_batch_size = max_batch_size
        self.conv = [
            torch.empty(
                (shape[0], max_batch_size, *shape[1:]),
                dtype=conv_dtype,
                device=self.device,
            )
            for shape in conv_shapes
        ]
        self.temporal = torch.empty(
            (temporal_shape[0], max_batch_size, *temporal_shape[1:]),
            dtype=temporal_dtype,
            device=self.device,
        )
        self.kernel_indices = torch.arange(
            max_batch_size, dtype=torch.long, device=self.device
        )

    def load(self, conv: list[torch.Tensor], temporal: torch.Tensor) -> GraphStateBatch:
        """Copy compact state into stable graph buffers and return the active slice."""
        batch_size = temporal.shape[1]
        if batch_size > self.max_batch_size:
            raise ValueError("state batch exceeds graph buffer capacity")
        if len(conv) != len(self.conv):
            raise ValueError("conv layer count does not match graph buffer")
        for target, source in zip(self.conv, conv):
            if target.shape[0] != source.shape[0] or target.shape[2:] != source.shape[2:]:
                raise ValueError("conv state shape does not match graph buffer")
            target[:, :batch_size].copy_(source)
        if self.temporal.shape[0] != temporal.shape[0] or self.temporal.shape[2:] != temporal.shape[2:]:
            raise ValueError("temporal state shape does not match graph buffer")
        self.temporal[:, :batch_size].copy_(temporal)
        return GraphStateBatch(
            conv=[target[:, :batch_size] for target in self.conv],
            temporal=self.temporal[:, :batch_size],
            kernel_indices=self.kernel_indices[:batch_size],
        )
