"""Slab-backed logical Mamba state storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class MambaStateSlot:
    slab: int
    offset: int


class MambaStateStore:
    """Growable state fields with stable logical slot ids.

    Two slot-mapping modes:

    - identity (default): physical slot == logical id. Capacity therefore tracks
      the highest logical id ever bound. This is what the contiguous-pool
      compatibility helpers (``from_state_tensors`` / ``import_fields``) rely on,
      since they preload a full pool row-for-row.
    - ``reuse_released_slots=True``: physical slots are assigned from a free list
      and returned to it on ``release``. Capacity then tracks *concurrently live*
      slots rather than the id range, which is what gives fixed occupancy — the
      slot allocator hands out logical ids from the whole ``1..size`` space, so
      under identity mapping a long-running server would eventually fault in
      every slab regardless of how few requests are actually in flight.
    """

    def __init__(
        self,
        field_shapes: dict[str, tuple[int, ...]],
        *,
        capacity: int,
        slab_size: int,
        dtype: dict[str, torch.dtype],
        device: torch.device | str,
        reuse_released_slots: bool = False,
        max_capacity: int | None = None,
    ) -> None:
        if capacity < 0 or slab_size <= 0:
            raise ValueError("capacity must be non-negative and slab_size positive")
        if not field_shapes or set(field_shapes) != set(dtype):
            raise ValueError("field_shapes and dtype must have the same non-empty keys")
        if max_capacity is not None and max_capacity < capacity:
            raise ValueError("max_capacity must be at least the initial capacity")
        self.device = torch.device(device)
        self.slab_size = slab_size
        self._capacity = 0
        self._slots: dict[int, MambaStateSlot] = {}
        self._reuse_released_slots = reuse_released_slots
        self._max_capacity = max_capacity
        # Physical slots returned by ``release`` (reuse mode only).
        self._free_physical: list[int] = []
        # Highest physical slot handed out so far (reuse mode). Tracked apart
        # from ``_capacity``, which counts *allocated* rows: pre-allocating a
        # slab must not make the next bind skip past it.
        # Physical row 0 mirrors the contiguous pool's slot 0: a permanently
        # zeroed row that padded CUDA-graph rows read from. Never handed out.
        self.pad_row = 0
        self._next_physical = 1 if reuse_released_slots else 0
        # Device-side logical->physical map. Maintained on bind/release (always
        # outside a CUDA graph), so a captured forward can translate slot ids
        # with a pure device gather instead of host-side list work.
        self._slot_to_physical = (
            torch.zeros(
                (max_capacity or capacity or 1,), dtype=torch.int32, device=self.device
            )
            if reuse_released_slots
            else None
        )
        # Logical ids released but not yet rebound (identity mode only); under
        # identity mapping the slot is reclaimed implicitly by rebinding the id.
        self._free_logical: set[int] = set()
        self._slabs: dict[str, list[torch.Tensor]] = {
            name: [] for name in field_shapes
        }
        self._field_shapes = dict(field_shapes)
        self._dtype = dict(dtype)
        self.ensure_slots(capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def slab_count(self) -> int:
        return (self._capacity + self.slab_size - 1) // self.slab_size

    def ensure_slots(self, required: int) -> None:
        """Grow to cover ``required`` slots, one uniformly sized slab at a time.

        Slabs are always ``slab_size`` rows so that slot addressing stays exactly
        ``divmod(physical_id, slab_size)``. A partially sized slab would desync
        that mapping from the real layout and index the wrong row (or run off the
        end of the slab), so capacity rounds up to a slab multiple.
        """
        if required < 0:
            raise ValueError("required must be non-negative")
        while self._capacity < required:
            for name, shape in self._field_shapes.items():
                self._slabs[name].append(
                    torch.zeros(
                        (shape[0], self.slab_size, *shape[1:]),
                        dtype=self._dtype[name],
                        device=self.device,
                    )
                )
            self._capacity += self.slab_size

    def _alloc_physical(self) -> int:
        """Assign a physical slot in reuse mode, growing only when none is idle."""
        if self._free_physical:
            return self._free_physical.pop()
        physical_id = self._next_physical
        if self._max_capacity is not None and physical_id >= self._max_capacity:
            raise RuntimeError(
                f"Mamba state store exhausted: {self._max_capacity} concurrent "
                "slots in use. Raise the slab capacity or lower the number of "
                "concurrent requests."
            )
        # No-op when a pre-allocated slab already covers this row.
        self.ensure_slots(physical_id + 1)
        self._next_physical += 1
        return physical_id

    def bind(self, logical_ids: Iterable[int]) -> torch.Tensor:
        ids = [int(value) for value in logical_ids]
        if any(value < 0 for value in ids):
            raise IndexError("logical Mamba slot ids must be non-negative")
        if not self._reuse_released_slots:
            self.ensure_slots(max(ids, default=-1) + 1)
        physical = []
        for logical in ids:
            slot = self._slots.get(logical)
            if slot is None:
                # Identity mapping keeps the compat helpers row-aligned; reuse
                # mode draws from the free list so capacity tracks live slots.
                physical_id = (
                    self._alloc_physical() if self._reuse_released_slots else logical
                )
                slot = MambaStateSlot(
                    slab=physical_id // self.slab_size,
                    offset=physical_id % self.slab_size,
                )
                self._slots[logical] = slot
            # Rebinding a released id reclaims it (identity mode bookkeeping).
            self._free_logical.discard(logical)
            physical_id = slot.slab * self.slab_size + slot.offset
            if self._slot_to_physical is not None and logical < len(
                self._slot_to_physical
            ):
                self._slot_to_physical[logical] = physical_id
            physical.append(physical_id)
        return torch.tensor(physical, dtype=torch.long, device=self.device)

    def bind_padded(self, logical_ids: Iterable[int]) -> torch.Tensor:
        """Bind slot ids where negatives are CUDA-graph padding.

        Padded rows resolve to the reserved zero row instead of allocating, so a
        graph-sized batch does not consume slab capacity for rows it ignores.
        """
        ids = [int(v) for v in logical_ids]
        real = [v for v in ids if v >= 0]
        mapping = {}
        if real:
            bound = self.bind(real).tolist()
            mapping = dict(zip(real, bound))
        return torch.tensor(
            [mapping[v] if v >= 0 else self.pad_row for v in ids],
            dtype=torch.long,
            device=self.device,
        )

    def ensure_logical_slots(self, logical_ids: Iterable[int]) -> torch.Tensor:
        return self.bind(logical_ids)

    def field(self, name: str, physical_ids: torch.Tensor) -> torch.Tensor:
        if name not in self._slabs:
            raise KeyError(name)
        pieces = []
        for physical in physical_ids.flatten().tolist():
            slab, offset = divmod(int(physical), self.slab_size)
            pieces.append(self._slabs[name][slab][:, offset : offset + 1])
        if not pieces:
            shape = self._field_shapes[name]
            return torch.empty(
                (shape[0], 0, *shape[1:]),
                dtype=self._dtype[name],
                device=self.device,
            )
        return torch.cat(pieces, dim=1)

    @property
    def is_single_slab(self) -> bool:
        """Whether every slot lives in one slab, enabling the direct path."""
        return self.slab_count == 1

    @property
    def covers_all_slots(self) -> bool:
        """Whether one slab can address every slot the allocator may issue.

        The direct path indexes the slab with translated slot ids, so it is only
        safe when every issuable id has a distinct row. Radix caching allocates
        track and ping-pong slots across the whole range, and those are bound
        lazily -- with a smaller slab their translations collide on the reserved
        row or run past the end (illegal memory access).
        """
        return (
            self.is_single_slab
            and self._max_capacity is not None
            and self.capacity >= self._max_capacity
        )

    def slab_tensor(self, name: str) -> torch.Tensor:
        """The sole slab for a field, shaped like the contiguous pool's state.

        Only valid when ``is_single_slab``. Kernels can index this directly with
        physical slot ids -- no gather, no scatter, no allocation -- which is
        what makes a captured CUDA graph legal.
        """
        if not self.is_single_slab:
            raise RuntimeError("slab_tensor requires a single slab")
        return self._slabs[name][0]

    def translate_slots(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """Logical -> physical slot ids as a pure device gather.

        Negative (CUDA-graph padded) ids read the reserved zero row, matching
        the contiguous pool where padded rows point at slot 0.
        """
        if self._slot_to_physical is None:
            return slot_ids
        return self._slot_to_physical[slot_ids.clamp(min=0).long()].to(slot_ids.dtype)

    def field_shape(self, name: str) -> tuple[int, ...]:
        """Per-field shape as ``(layer_dim, *per_slot_dims)``."""
        if name not in self._field_shapes:
            raise KeyError(name)
        return self._field_shapes[name]

    def field_dtype(self, name: str) -> torch.dtype:
        """Storage dtype of a field, without materializing any rows."""
        if name not in self._dtype:
            raise KeyError(name)
        return self._dtype[name]

    def field_layer(
        self, name: str, layer: int, physical_ids: torch.Tensor
    ) -> torch.Tensor:
        """Gather one layer's rows for a batch as a compact ``[bs, ...]`` tensor.

        Kernels run per layer, so this avoids materializing every layer's state
        when only one is needed. The result is a fresh tensor; write it back with
        ``scatter_layer`` after the kernel has updated it in place.
        """
        if name not in self._slabs:
            raise KeyError(name)
        pieces = []
        for physical in physical_ids.flatten().tolist():
            slab, offset = divmod(int(physical), self.slab_size)
            pieces.append(self._slabs[name][slab][layer, offset : offset + 1])
        if not pieces:
            shape = self._field_shapes[name]
            return torch.empty(
                (0, *shape[1:]), dtype=self._dtype[name], device=self.device
            )
        return torch.cat(pieces, dim=0)

    def scatter_layer(
        self, name: str, layer: int, physical_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        ids = physical_ids.flatten().tolist()
        if values.shape[0] != len(ids):
            raise ValueError("values batch dimension does not match physical_ids")
        for index, physical in enumerate(ids):
            if self._reuse_released_slots and int(physical) == self.pad_row:
                # Padded CUDA-graph rows all alias the reserved zero row; writing
                # them back would poison it for every later replay.
                continue
            slab, offset = divmod(int(physical), self.slab_size)
            self._slabs[name][slab][layer, offset].copy_(values[index])

    def scatter(
        self, name: str, physical_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        ids = physical_ids.flatten().tolist()
        if values.shape[1] != len(ids):
            raise ValueError("values slot dimension does not match physical_ids")
        for index, physical in enumerate(ids):
            if self._reuse_released_slots and int(physical) == self.pad_row:
                continue
            slab, offset = divmod(int(physical), self.slab_size)
            self._slabs[name][slab][:, offset].copy_(values[:, index])

    def release(self, logical_ids: Iterable[int]) -> None:
        """Clear a logical slot and mark it idle without shrinking the slabs.

        Occupancy is fixed either way: the slabs keep their size and the slot is
        zeroed so a later bind starts clean. In reuse mode the physical slot goes
        back on the free list and is handed to the next unseen logical id; under
        identity mapping it is reclaimed by rebinding the same id.
        """
        ids = [int(value) for value in logical_ids]
        physical = []
        for logical in ids:
            slot = self._slots.pop(logical, None)
            if slot is not None:
                physical_id = slot.slab * self.slab_size + slot.offset
                physical.append(physical_id)
                if self._reuse_released_slots:
                    self._free_physical.append(physical_id)
                    if self._slot_to_physical is not None and logical < len(
                        self._slot_to_physical
                    ):
                        self._slot_to_physical[logical] = self.pad_row
                else:
                    self._free_logical.add(logical)
        if physical:
            self.clear(torch.tensor(physical, dtype=torch.long, device=self.device))

    def available_logical_slots(self) -> int:
        """Number of slots currently idle and immediately reusable."""
        if self._reuse_released_slots:
            return len(self._free_physical)
        return len(self._free_logical)

    @property
    def live_slots(self) -> int:
        """Slots currently bound to a logical id."""
        return len(self._slots)

    def state_fields(self) -> tuple[str, ...]:
        return tuple(self._slabs)

    def iter_field_slabs(self, name: str):
        if name not in self._slabs:
            raise KeyError(name)
        return iter(self._slabs[name])

    def copy_slots(self, source_ids: torch.Tensor, dest_ids: torch.Tensor) -> None:
        if source_ids.numel() != dest_ids.numel():
            raise ValueError("source and destination slot counts must match")
        for name in self._slabs:
            values = self.field(name, source_ids)
            self.scatter(name, dest_ids, values)

    def clear(self, physical_ids: torch.Tensor) -> None:
        for name in self._slabs:
            for physical in physical_ids.flatten().tolist():
                slab, offset = divmod(int(physical), self.slab_size)
                self._slabs[name][slab][:, offset].zero_()

    @classmethod
    def from_state_tensors(
        cls, fields: dict[str, torch.Tensor], *, slab_size: int
    ) -> "MambaStateStore":
        if not fields:
            raise ValueError("fields cannot be empty")
        capacity = next(iter(fields.values())).shape[1]
        if any(tensor.shape[1] != capacity for tensor in fields.values()):
            raise ValueError("all state fields must have the same slot capacity")
        return cls(
            {name: (tensor.shape[0], *tensor.shape[2:]) for name, tensor in fields.items()},
            capacity=capacity,
            slab_size=slab_size,
            dtype={name: tensor.dtype for name, tensor in fields.items()},
            device=next(iter(fields.values())).device,
        )

    def import_fields(self, fields: dict[str, torch.Tensor]) -> None:
        if set(fields) != set(self._slabs):
            raise ValueError("state field names do not match the store")
        capacity = next(iter(fields.values())).shape[1]
        self.ensure_slots(capacity)
        ids = torch.arange(capacity, dtype=torch.long, device=self.device)
        for name, tensor in fields.items():
            self.scatter(name, ids, tensor)
