"""Quantization-aware expert weight residency primitives.

This module deliberately does not know about a particular MoE kernel.  A quantization
method can provide one tensor bundle per local expert after its normal checkpoint
loader and post-processing have completed.  The residency manager then keeps those
bundles in host storage and materializes only the requested experts for a forward.

The first implementation is synchronous and single-process.  It is a foundation for
MuxMoE rather than a replacement for the existing dispatcher or runner backends.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping, Sequence

import torch


class ExpertWeightBundle:
    """All tensors required to execute one already-processed local expert.

    ``weights`` must contain the complete quantization method contract for the
    expert (for example ``w13_weight``, ``w2_weight`` and their scale tensors).
    Keeping the bundle opaque prevents an offload layer from accidentally dropping
    metadata required by a quantized kernel.
    """

    def __init__(
        self,
        weights: Mapping[str, torch.Tensor],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not weights:
            raise ValueError("An expert weight bundle cannot be empty")
        self.weights = dict(weights)
        self.metadata = dict(metadata or {})
        self._validate()

    def _validate(self) -> None:
        for name, tensor in self.weights.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expert tensor {name!r} is not a torch.Tensor")
            if tensor.ndim == 0:
                raise ValueError(f"Expert tensor {name!r} must have an expert shape")
            if tensor.device.type != "cpu":
                raise ValueError(
                    f"Expert tensor {name!r} must be stored on CPU, got {tensor.device}"
                )

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for tensor in self.weights.values()
        )


class ResidentExpertBatch:
    """Compact tensors and ID remapping returned by :class:`ExpertResidencyManager`."""

    def __init__(
        self,
        *,
        expert_ids: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        id_to_slot: Mapping[int, int],
    ) -> None:
        self.expert_ids = expert_ids
        self.weights = dict(weights)
        self.id_to_slot = dict(id_to_slot)


class ExpertResidencyManager:
    """Synchronous, layer-local expert storage with bounded resident slots.

    The manager operates on bundles after the model's quantization loader has
    finished all repacking and scale construction.  It therefore never infers a
    quantization format from checkpoint names and never reconstructs a packed
    tensor from a gate/up/down convention.
    """

    def __init__(self, *, max_resident_experts: int) -> None:
        if max_resident_experts <= 0:
            raise ValueError("max_resident_experts must be positive")
        self.max_resident_experts = max_resident_experts
        self._layers: dict[int, tuple[ExpertWeightBundle, ...]] = {}
        self._cache: dict[tuple[int, str], OrderedDict[int, int]] = {}
        self._device_slots: dict[
            tuple[int, str], dict[str, torch.Tensor]
        ] = {}
        self._host_bytes = 0

    @property
    def host_bytes(self) -> int:
        return self._host_bytes

    @property
    def cached_expert_count(self) -> int:
        """Number of experts currently resident across all layer/device caches."""
        return sum(len(cache) for cache in self._cache.values())

    def register_layer(
        self, layer_id: int, expert_bundles: Sequence[ExpertWeightBundle]
    ) -> None:
        if not expert_bundles:
            raise ValueError(f"Layer {layer_id} has no expert bundles")
        if layer_id in self._layers:
            raise ValueError(f"Layer {layer_id} is already registered")
        names = set(expert_bundles[0].weights)
        for expert_id, bundle in enumerate(expert_bundles):
            if set(bundle.weights) != names:
                raise ValueError(
                    f"Layer {layer_id} expert {expert_id} has a different tensor contract"
                )
            for name in names:
                reference = expert_bundles[0].weights[name]
                tensor = bundle.weights[name]
                if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                    raise ValueError(
                        f"Layer {layer_id} expert {expert_id} tensor {name!r} "
                        "does not match the layer contract"
                    )
        self._layers[layer_id] = tuple(expert_bundles)
        self._host_bytes += sum(bundle.nbytes for bundle in expert_bundles)

    def ensure_resident(
        self,
        layer_id: int,
        expert_ids: Iterable[int] | torch.Tensor,
        *,
        device: torch.device | str = "cpu",
    ) -> ResidentExpertBatch:
        """Return requested experts in compact slot order.

        The current implementation copies bundles synchronously and is intended
        for correctness bring-up.  A CUDA caller must provide a device; asynchronous
        streams and eviction policy are deliberately left for the next phase.
        """
        bundles = self._layers.get(layer_id)
        if bundles is None:
            raise KeyError(f"Layer {layer_id} is not registered")
        requested = [
            int(value) for value in torch.as_tensor(expert_ids).flatten().tolist()
        ]
        unique_ids = list(dict.fromkeys(requested))
        if len(unique_ids) > self.max_resident_experts:
            raise RuntimeError(
                f"Layer {layer_id} requests {len(unique_ids)} experts, exceeding "
                f"the residency limit {self.max_resident_experts}"
            )
        if any(expert_id < 0 or expert_id >= len(bundles) for expert_id in unique_ids):
            raise IndexError(f"Layer {layer_id} contains an out-of-range expert id")

        target = torch.device(device)
        cache_key = (layer_id, str(target))
        cache = self._cache.setdefault(cache_key, OrderedDict())
        slots = self._device_slots.setdefault(cache_key, {})
        materialized: dict[str, list[torch.Tensor]] = {}
        id_to_slot: dict[int, int] = {}
        for requested_slot, expert_id in enumerate(unique_ids):
            bundle = bundles[expert_id]
            if expert_id in cache:
                slot = cache[expert_id]
                cache.move_to_end(expert_id)
                id_to_slot[expert_id] = slot
                for name, tensor in slots.items():
                    materialized.setdefault(name, []).append(tensor[slot])
                continue

            used_slots = set(cache.values())
            free_slots = [
                candidate
                for candidate in range(self.max_resident_experts)
                if candidate not in used_slots
            ]
            if not free_slots:
                evicted_id, slot = cache.popitem(last=False)
                del evicted_id
            else:
                slot = free_slots[0]
            id_to_slot[expert_id] = slot
            for name, tensor in bundle.weights.items():
                slot_tensor = slots.get(name)
                if slot_tensor is None:
                    slot_tensor = torch.empty(
                        (self.max_resident_experts, *tensor.shape),
                        dtype=tensor.dtype,
                        device=target,
                    )
                    slots[name] = slot_tensor
                slot_tensor[slot].copy_(tensor, non_blocking=target.type == "cuda")
                materialized.setdefault(name, []).append(slot_tensor[slot])
            cache[expert_id] = slot
            cache.move_to_end(expert_id)

        self._evict_if_needed(cache_key)
        weights = {
            name: torch.stack(tensors, dim=0) for name, tensors in materialized.items()
        }
        return ResidentExpertBatch(
            expert_ids=torch.tensor(unique_ids, dtype=torch.long, device=target),
            weights=weights,
            id_to_slot=id_to_slot,
        )

    def _evict_if_needed(self, cache_key: tuple[int, str]) -> None:
        cache = self._cache[cache_key]
        while len(cache) > self.max_resident_experts:
            cache.popitem(last=False)
