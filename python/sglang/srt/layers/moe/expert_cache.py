"""Host-backed cache for routed-expert payloads.

The cache deliberately owns only routed-expert tensors.  Router, attention,
shared-expert, and activation-scale parameters remain on the accelerator.  A
layer is admitted to host memory atomically: either its complete expert payload
fits the host budget or the layer remains a normal device-resident MoE layer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, NamedTuple, Optional, Sequence

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)

logger = logging.getLogger(__name__)

_HOST_HEADROOM_LAYERS = 3
_PREFETCH_MIN_SAMPLES = 128
_PREFETCH_MIN_HIT_RATE = 0.80
_CUDA_MEMCPY_BATCH_MIN_VERSION = 12080


class _CudaBatchCopy:
    """Lazy CUDA runtime wrapper for independent pointer-to-pointer copies."""

    def __init__(self) -> None:
        self._runtime = None
        self._attributes = {}
        self._available: Optional[bool] = None

    def _initialize(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import cuda.bindings.runtime as runtime
        except ImportError:
            self._available = False
            return False

        err, version = runtime.cudaRuntimeGetVersion()
        if (
            err != runtime.cudaError_t.cudaSuccess
            or version < _CUDA_MEMCPY_BATCH_MIN_VERSION
            or not hasattr(runtime, "cudaMemcpyBatchAsync")
        ):
            self._available = False
            return False

        attributes = {}
        for prefer_overlap in (False, True):
            item = runtime.cudaMemcpyAttributes()
            item.srcAccessOrder = (
                runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
            )
            if prefer_overlap:
                item.flags = int(
                    runtime.cudaMemcpyFlags.cudaMemcpyFlagPreferOverlapWithCompute
                )
            attributes[prefer_overlap] = item
        self._runtime = runtime
        self._attributes = attributes
        self._available = True
        return True

    @staticmethod
    def _compatible(copies: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> bool:
        for destination, source in copies:
            if (
                destination.device.type != "cuda"
                or source.device.type not in ("cpu", "cuda")
                or (source.device.type == "cpu" and not source.is_pinned())
                or destination.dtype != source.dtype
                or destination.numel() != source.numel()
                or not destination.is_contiguous()
                or not source.is_contiguous()
            ):
                return False
        return True

    def submit(
        self,
        copies: Sequence[tuple[torch.Tensor, torch.Tensor]],
        stream: torch.cuda.Stream,
        prefer_overlap: bool = False,
    ) -> bool:
        if len(copies) < 2 or not self._compatible(copies) or not self._initialize():
            return False

        runtime = self._runtime
        err = runtime.cudaMemcpyBatchAsync(
            [destination.data_ptr() for destination, _ in copies],
            [source.data_ptr() for _, source in copies],
            [tensor_nbytes(source) for _, source in copies],
            len(copies),
            [self._attributes[prefer_overlap]],
            [0],
            1,
            stream.cuda_stream,
        )[0]
        if err == runtime.cudaError_t.cudaSuccess:
            return True

        unsupported_errors = {
            getattr(runtime.cudaError_t, "cudaErrorNotSupported", None),
            getattr(runtime.cudaError_t, "cudaErrorCallRequiresNewerDriver", None),
        }
        if err in unsupported_errors:
            self._available = False
            logger.warning(
                "CUDA batched memcpy is unavailable (%s); falling back to "
                "individual asynchronous copies.",
                err,
            )
            return False
        raise RuntimeError(f"cudaMemcpyBatchAsync failed with {err}")


_CUDA_BATCH_COPY = _CudaBatchCopy()


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def shape_nbytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


@dataclass(frozen=True)
class ExpertPayloadSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return shape_nbytes(self.shape, self.dtype)

    @property
    def expert_nbytes(self) -> int:
        return shape_nbytes(self.shape[1:], self.dtype)


def modelopt_nvfp4_payload_specs(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    is_gated: bool,
) -> tuple[ExpertPayloadSpec, ...]:
    """Return the final CUTLASS payload shapes derived from tensor metadata."""
    if hidden_size % 2 or hidden_size % group_size:
        raise ValueError(
            f"NVFP4 hidden_size={hidden_size} must be divisible by 2 and "
            f"group_size={group_size}"
        )
    if intermediate_size % 2 or intermediate_size % group_size:
        raise ValueError(
            f"NVFP4 intermediate_size={intermediate_size} must be divisible by 2 "
            f"and group_size={group_size}"
        )
    num_shards = 2 if is_gated else 1
    w13_rows = num_shards * intermediate_size
    w2_rows = hidden_size
    w13_scale_rows = math.ceil(w13_rows / 128) * 128
    w13_scale_cols = math.ceil((hidden_size // group_size) / 4) * 4
    w2_scale_rows = math.ceil(w2_rows / 128) * 128
    intermediate_pad = w13_scale_rows - w13_rows
    if is_gated and intermediate_pad:
        raise ValueError(
            "gated ModelOpt NVFP4 CUTLASS MoE requires the fused intermediate "
            f"dimension to be 128-aligned, got {w13_rows}"
        )
    final_intermediate = intermediate_size + intermediate_pad
    w2_scale_cols = math.ceil((final_intermediate // group_size) / 4) * 4

    # process_weights_after_loading pads packed weights to the swizzled scale
    # extent.  FP4 packs two input elements in each byte.
    w13_weight_rows = w13_scale_rows
    w2_weight_cols = final_intermediate // 2
    return (
        ExpertPayloadSpec(
            "w13_weight",
            (num_experts, w13_weight_rows, hidden_size // 2),
            torch.uint8,
        ),
        ExpertPayloadSpec(
            "w2_weight",
            (num_experts, hidden_size, w2_weight_cols),
            torch.uint8,
        ),
        ExpertPayloadSpec(
            "w13_blockscale_swizzled",
            (num_experts, w13_scale_rows, w13_scale_cols),
            torch.float8_e4m3fn,
        ),
        ExpertPayloadSpec(
            "w2_blockscale_swizzled",
            (num_experts, w2_scale_rows, w2_scale_cols),
            torch.float8_e4m3fn,
        ),
        ExpertPayloadSpec("g1_alphas", (num_experts,), torch.float32),
        ExpertPayloadSpec("g2_alphas", (num_experts,), torch.float32),
    )


def moe_wna16_payload_specs(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    weight_bits: int,
    group_size: int,
    has_zp: bool,
    dtype: torch.dtype,
) -> tuple[ExpertPayloadSpec, ...]:
    """Return the final Triton WNA16 payload shapes."""
    if weight_bits not in (4, 8):
        raise ValueError(f"WNA16 expert cache requires 4 or 8 bits, got {weight_bits}")
    if group_size <= 0:
        raise ValueError(f"WNA16 group_size must be positive, got {group_size}")
    pack_factor = 8 // weight_bits
    if hidden_size % pack_factor or intermediate_size % pack_factor:
        raise ValueError(
            "WNA16 packed dimensions must be divisible by the uint8 pack factor"
        )
    if hidden_size % group_size or intermediate_size % group_size:
        raise ValueError(
            "WNA16 hidden and intermediate dimensions must be divisible by "
            f"group_size={group_size}"
        )

    specs = [
        ExpertPayloadSpec(
            "w13_qweight",
            (num_experts, 2 * intermediate_size, hidden_size // pack_factor),
            torch.uint8,
        ),
        ExpertPayloadSpec(
            "w2_qweight",
            (num_experts, hidden_size, intermediate_size // pack_factor),
            torch.uint8,
        ),
        ExpertPayloadSpec(
            "w13_scales",
            (num_experts, 2 * intermediate_size, hidden_size // group_size),
            dtype,
        ),
        ExpertPayloadSpec(
            "w2_scales",
            (num_experts, hidden_size, intermediate_size // group_size),
            dtype,
        ),
    ]
    if has_zp:
        specs.extend(
            (
                ExpertPayloadSpec(
                    "w13_qzeros",
                    (
                        num_experts,
                        2 * intermediate_size // pack_factor,
                        hidden_size // group_size,
                    ),
                    torch.uint8,
                ),
                ExpertPayloadSpec(
                    "w2_qzeros",
                    (
                        num_experts,
                        hidden_size // pack_factor,
                        intermediate_size // group_size,
                    ),
                    torch.uint8,
                ),
            )
        )
    return tuple(specs)


def select_complete_layers(
    layer_sizes: Sequence[int], host_budget_bytes: int
) -> list[int]:
    """Select a prefix of complete layers without exceeding the host budget."""
    if host_budget_bytes < 0:
        raise ValueError("host_budget_bytes must be non-negative")
    selected = []
    used = 0
    for index, size in enumerate(layer_sizes):
        if size <= 0:
            raise ValueError(f"layer payload size must be positive, got {size}")
        if used + size > host_budget_bytes:
            break
        selected.append(index)
        used += size
    return selected


class CacheAccess(NamedTuple):
    hits: dict[int, int]
    admissions: dict[int, int]
    evictions: tuple[int, ...]
    transient: tuple[int, ...]


@dataclass
class _SlotState:
    expert: Optional[int] = None
    last_used: int = -1
    lease_until: int = -1


class ExpertCachePolicy:
    """Pure-Python lease/LRU metadata policy, independent of CUDA tensors."""

    def __init__(self, capacity: int, policy: str = "lease", lease: int = 3):
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        if policy not in ("lease", "lru"):
            raise ValueError(f"unsupported expert cache policy: {policy!r}")
        if lease < 0:
            raise ValueError("lease must be non-negative")
        self.capacity = capacity
        self.policy = policy
        self.lease = lease
        self.step = 0
        self.slots = [_SlotState() for _ in range(capacity)]
        self.expert_to_slot: dict[int, int] = {}

    def access(
        self,
        experts: Iterable[int],
        *,
        prefill: bool = False,
        prefill_reuse: bool = True,
        allow_lease_break: bool = False,
    ) -> CacheAccess:
        requested = tuple(dict.fromkeys(int(expert) for expert in experts))
        protected = set(requested)
        hits: dict[int, int] = {}
        admissions: dict[int, int] = {}
        evictions: list[int] = []
        transient: list[int] = []

        if prefill and not prefill_reuse:
            return CacheAccess({}, {}, (), requested)

        if not prefill:
            self.step += 1

        for expert in requested:
            slot = self.expert_to_slot.get(expert)
            if slot is None:
                continue
            hits[expert] = slot
            if not prefill:
                state = self.slots[slot]
                state.last_used = self.step
                state.lease_until = self.step + self.lease

        # Prefill may read existing residents but never admits, evicts, or
        # renews a lease, preventing a long prompt from polluting decode state.
        if prefill:
            transient.extend(expert for expert in requested if expert not in hits)
            return CacheAccess(hits, {}, (), tuple(transient))

        for expert in requested:
            if expert in hits:
                continue
            empty_slot = next(
                (i for i, state in enumerate(self.slots) if state.expert is None),
                None,
            )
            slot = empty_slot
            if slot is None:
                candidates = []
                for i, state in enumerate(self.slots):
                    if state.expert in protected:
                        continue
                    if self.policy == "lease" and state.lease_until > self.step:
                        continue
                    candidates.append((state.last_used, i))
                if not candidates and allow_lease_break:
                    candidates = [
                        (state.last_used, i)
                        for i, state in enumerate(self.slots)
                        if state.expert not in protected
                    ]
                if candidates:
                    _, slot = min(candidates)
            if slot is None:
                transient.append(expert)
                continue

            state = self.slots[slot]
            if state.expert is not None:
                evictions.append(state.expert)
                del self.expert_to_slot[state.expert]
            state.expert = expert
            state.last_used = self.step
            state.lease_until = self.step + self.lease
            self.expert_to_slot[expert] = slot
            admissions[expert] = slot

        return CacheAccess(hits, admissions, tuple(evictions), tuple(transient))


def apply_miss_only_route_drop(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    resident_experts: Iterable[int],
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop only low-weight misses, preserve top-1, and renormalize per token.

    Returns ``(weights, ids, dropped_mask)``.  A dropped lane is assigned its
    token's top-1 expert with zero weight so downstream kernels never observe an
    invalid compact expert id.
    """
    if not 0.0 <= threshold < 1.0:
        raise ValueError("route drop threshold must be in [0, 1)")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape")
    if topk_weights.ndim != 2:
        raise ValueError("top-k routing tensors must be two-dimensional")

    resident = tuple(int(expert) for expert in resident_experts)
    if resident:
        resident_tensor = torch.tensor(
            resident, dtype=topk_ids.dtype, device=topk_ids.device
        )
        hit_mask = torch.isin(topk_ids, resident_tensor)
    else:
        hit_mask = torch.zeros_like(topk_ids, dtype=torch.bool)

    dropped = (~hit_mask) & (topk_weights < threshold)
    top1_index = topk_weights.argmax(dim=-1, keepdim=True)
    dropped.scatter_(1, top1_index, False)

    kept_weights = topk_weights.masked_fill(dropped, 0)
    denominator = kept_weights.sum(dim=-1, keepdim=True)
    if torch.any(denominator <= 0):
        raise ValueError("route weights must retain positive mass per token")
    kept_weights = kept_weights / denominator
    top1_ids = topk_ids.gather(1, top1_index).expand_as(topk_ids)
    kept_ids = torch.where(dropped, top1_ids, topk_ids)
    return kept_weights, kept_ids, dropped


def remap_expert_ids(
    expert_ids: torch.Tensor, expert_to_compact: Mapping[int, int]
) -> torch.Tensor:
    """Map every expert ID to a compact kernel-local ID, failing closed."""
    remapped = torch.full_like(expert_ids, -1)
    for expert, compact in expert_to_compact.items():
        remapped.masked_fill_(expert_ids == expert, compact)
    if torch.any(remapped < 0):
        missing = torch.unique(expert_ids[remapped < 0]).to("cpu").tolist()
        raise ValueError(f"missing compact IDs for experts {missing}")
    return remapped


def remap_single_token_expert_ids(
    expert_ids: torch.Tensor,
    expert_ids_cpu: Sequence[int],
    expert_to_compact: Mapping[int, int],
) -> torch.Tensor:
    """Remap a single token from its already-synchronized CPU expert IDs."""
    if expert_ids.ndim != 2 or expert_ids.shape[0] != 1:
        raise ValueError("single-token expert remapping requires shape [1, top_k]")
    if len(expert_ids_cpu) != expert_ids.shape[1]:
        raise ValueError("CPU expert IDs must match the single-token top-k width")
    try:
        compact = [expert_to_compact[int(expert)] for expert in expert_ids_cpu]
    except KeyError as exc:
        raise ValueError(f"missing compact ID for expert {exc.args[0]}") from exc
    return expert_ids.new_tensor(compact).view_as(expert_ids)


@dataclass
class ExpertCacheStats:
    hits: int = 0
    misses: int = 0
    h2d_bytes: int = 0
    d2d_bytes: int = 0
    admissions: int = 0
    evictions: int = 0
    transient: int = 0
    route_drops: int = 0
    dropped_mass: float = 0.0
    prefill_reuse: int = 0
    prefetch_h2d_bytes: int = 0
    prefetch_selected: int = 0
    prefetch_admissions: int = 0
    prefetch_evictions: int = 0
    prefetch_predictions: int = 0
    prefetch_matches: int = 0
    prefetch_useful: int = 0
    prefetch_wasted: int = 0
    staging_peak_bytes: int = 0
    batched_copy_submissions: int = 0
    batched_copy_operations: int = 0
    fallback_copy_operations: int = 0
    forwards: int = 0

    def add(self, other: "ExpertCacheStats") -> None:
        for name in self.__dataclass_fields__:
            if name == "staging_peak_bytes":
                value = max(getattr(self, name), getattr(other, name))
            else:
                value = getattr(self, name) + getattr(other, name)
            setattr(self, name, value)


@dataclass
class PreparedExpertPayload:
    dispatch_output: object
    payload: Mapping[str, torch.Tensor]
    layer_cache: "LayerExpertCache"
    used_slots: tuple[int, ...]
    uses_staging: bool

    def mark_kernel_launched(self) -> None:
        self.layer_cache.mark_kernel_launched(
            used_slots=self.used_slots, uses_staging=self.uses_staging
        )


@dataclass
class LayerExpertCache:
    layer_id: int
    host_payload: dict[str, torch.Tensor]
    num_experts: int
    policy_name: str
    lease: int
    manager: "MoeExpertCacheManager"
    payload_format: str = "unknown"
    capacity: int = 0
    resident_payload: dict[str, torch.Tensor] = field(default_factory=dict)
    policy: ExpertCachePolicy = field(init=False)
    slot_events: list[Optional[torch.cuda.Event]] = field(default_factory=list)
    stats: ExpertCacheStats = field(default_factory=ExpertCacheStats)
    predicted_experts: set[int] = field(default_factory=set)
    predicted_order: tuple[int, ...] = ()
    prefetched_experts: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.policy = ExpertCachePolicy(0, self.policy_name, self.lease)

    @property
    def expert_bytes(self) -> int:
        return sum(tensor_nbytes(tensor[0]) for tensor in self.host_payload.values())

    @property
    def layer_bytes(self) -> int:
        return self.expert_bytes * self.num_experts

    def allocate(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.policy = ExpertCachePolicy(capacity, self.policy_name, self.lease)
        self.resident_payload = {
            name: torch.empty(
                (capacity, *tensor.shape[1:]), dtype=tensor.dtype, device=device
            )
            for name, tensor in self.host_payload.items()
        }
        self.slot_events = [None] * capacity

    def mark_kernel_launched(
        self, *, used_slots: tuple[int, ...], uses_staging: bool
    ) -> None:
        stream = torch.cuda.current_stream()
        event = torch.cuda.Event()
        event.record(stream)
        for slot in used_slots:
            self.slot_events[slot] = event
        if uses_staging:
            self.manager.staging_event = event


class MoeExpertCacheManager:
    """Process-local owner of host layers, GPU slabs, and shared staging."""

    def __init__(
        self,
        *,
        host_budget_bytes: int,
        gpu_budget_bytes: int,
        policy: str,
        lease: int,
        drop_threshold: float,
        prefill_cache_reuse: bool,
        cross_layer_prefetch: bool = True,
        prefetch_min_ratio: float = 0.5,
        resident_shallow_layers: int = 0,
    ):
        if not 0.0 < prefetch_min_ratio <= 1.0:
            raise ValueError("prefetch_min_ratio must be in (0, 1]")
        self.host_budget_bytes = host_budget_bytes
        self.gpu_budget_bytes = gpu_budget_bytes
        self.policy = policy
        self.lease = lease
        self.drop_threshold = drop_threshold
        self.prefill_cache_reuse = prefill_cache_reuse
        self.cross_layer_prefetch = cross_layer_prefetch
        self.prefetch_min_ratio = prefetch_min_ratio
        self.resident_shallow_layers = resident_shallow_layers
        self.reserved_host_bytes = 0
        self.reservation_headroom_bytes = 0
        self._logged_reservation_limit = False
        self.layers: list[LayerExpertCache] = []
        self.staging: dict[str, torch.Tensor] = {}
        self.staging_event: Optional[torch.cuda.Event] = None
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self._next_layer: dict[int, LayerExpertCache] = {}
        self._next_router: dict[int, torch.nn.Module] = {}
        self._pending_prefetch: dict[int, tuple[int, ...]] = {}
        self._prefetch_rank_attempts: list[int] = []
        self._prefetch_rank_hits: list[int] = []
        self.stats = ExpertCacheStats()
        self._host_attached = False
        self._finalized = False

    @property
    def enabled(self) -> bool:
        return self.host_budget_bytes > 0

    def reserve_layer(
        self,
        layer: torch.nn.Module,
        specs: Sequence[ExpertPayloadSpec],
        *,
        payload_format: str = "modelopt_nvfp4",
    ) -> bool:
        if not self.enabled:
            return False
        layer_id = int(getattr(layer, "layer_id", -1))
        if 0 <= layer_id < self.resident_shallow_layers:
            return False
        payload_bytes = sum(spec.nbytes for spec in specs)
        if payload_bytes > self.gpu_budget_bytes:
            raise ValueError(
                "--moe-expert-cache-gb is too small for one complete compact "
                f"staging layer: need {payload_bytes / 2**30:.3f} GiB, "
                f"budget is {self.gpu_budget_bytes / 2**30:.3f} GiB"
            )
        # Keep several complete maximum-size layers unused inside the user
        # budget.  The mmap source, pageable-to-pinned conversion, scheduler,
        # and tokenizer processes all need headroom in addition to the finalized
        # payload.  Consuming the nominal budget can leave the complete server
        # permanently above memory.high and turn small expert tensors into
        # repeated disk reads.
        candidate_headroom = max(
            self.reservation_headroom_bytes,
            _HOST_HEADROOM_LAYERS * payload_bytes,
        )
        if self.host_budget_bytes < payload_bytes + candidate_headroom:
            candidate_headroom = 0
        if (
            self.reserved_host_bytes + payload_bytes + candidate_headroom
            > self.host_budget_bytes
        ):
            if not self._logged_reservation_limit:
                logger.info(
                    "MoE expert offload reservation stopped at %.3f GiB; "
                    "retaining %.3f GiB (%d complete layers) of the %.3f GiB "
                    "host budget as load/runtime headroom.",
                    self.reserved_host_bytes / 2**30,
                    candidate_headroom / 2**30,
                    _HOST_HEADROOM_LAYERS,
                    self.host_budget_bytes / 2**30,
                )
                self._logged_reservation_limit = True
            return False
        self.reservation_headroom_bytes = candidate_headroom
        self.reserved_host_bytes += payload_bytes
        layer._moe_expert_offload_reserved = True
        layer._moe_expert_offload_expected_bytes = payload_bytes
        layer._moe_expert_offload_payload_names = tuple(spec.name for spec in specs)
        layer._moe_expert_offload_payload_format = payload_format
        return True

    def attach_model(self, model: torch.nn.Module, device: torch.device) -> None:
        if not self.enabled or self._host_attached:
            return
        for module in model.modules():
            if not getattr(module, "_moe_expert_offload_reserved", False):
                continue
            payload = {}
            payload_names = module._moe_expert_offload_payload_names
            payload_format = module._moe_expert_offload_payload_format
            for name in payload_names:
                tensor = getattr(module, name, None)
                if tensor is None:
                    raise RuntimeError(
                        f"offloaded {payload_format} MoE layer is missing payload "
                        f"{name!r}"
                    )
                tensor = (
                    tensor.data
                    if isinstance(tensor, torch.nn.Parameter)
                    else tensor
                )
                if tensor.device.type != "cpu":
                    raise RuntimeError(
                        f"offloaded payload {name} must be on CPU after loading, "
                        f"got {tensor.device}"
                    )
                if not tensor.is_pinned():
                    pinned = torch.empty_strided(
                        tensor.size(),
                        tensor.stride(),
                        dtype=tensor.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    pinned.copy_(tensor)
                    tensor = pinned
                    attr = getattr(module, name)
                    if isinstance(attr, torch.nn.Parameter):
                        attr.data = tensor
                    else:
                        setattr(module, name, tensor)
                payload[name] = tensor

            actual_bytes = sum(tensor_nbytes(tensor) for tensor in payload.values())
            expected_bytes = module._moe_expert_offload_expected_bytes
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"{payload_format} expert payload size changed after processing: "
                    f"expected {expected_bytes} bytes, got {actual_bytes} bytes"
                )
            layer_id = getattr(module, "layer_id", len(self.layers))
            cache = LayerExpertCache(
                layer_id=layer_id,
                host_payload=payload,
                num_experts=next(iter(payload.values())).shape[0],
                policy_name=self.policy,
                lease=self.lease,
                manager=self,
                payload_format=payload_format,
            )
            module._moe_expert_cache = cache
            self.layers.append(cache)

            if payload_format == "modelopt_nvfp4":
                self._finalize_modelopt_nvfp4_payload(module, device)

        if not self.layers:
            raise RuntimeError(
                "--moe-expert-offload-gb was enabled, but no supported MoE "
                "expert payload layer was reserved"
            )
        cache_by_experts = {
            id(module): getattr(module, "_moe_expert_cache")
            for module in model.modules()
            if getattr(module, "_moe_expert_cache", None) is not None
        }
        router_by_cache = {}
        for module in model.modules():
            experts = getattr(module, "experts", None)
            cache = cache_by_experts.get(id(experts))
            if cache is not None and hasattr(module, "gate") and hasattr(module, "topk"):
                router_by_cache[id(cache)] = module
                module._moe_cross_layer_prefetch_manager = self
        self._next_layer = {
            id(current): following
            for current, following in zip(self.layers, self.layers[1:])
        }
        self._next_router = {
            id(current): router_by_cache[id(following)]
            for current, following in zip(self.layers, self.layers[1:])
            if id(following) in router_by_cache
        }
        self._host_attached = True

    @staticmethod
    def _finalize_modelopt_nvfp4_payload(
        module: torch.nn.Module, device: torch.device
    ) -> None:
        # Global activation scales are deliberately device-resident.  The
        # loader's device context moves newly-derived parameters back with an
        # offloaded layer, so restore just these two scalars here.
        for name in ("w13_input_scale_quant", "w2_input_scale_quant"):
            param = getattr(module, name)
            if param.device.type == "cpu":
                param.data = param.data.to(device)

        # These source tensors are no longer read after the final CUTLASS
        # payload has been validated.  Dynamic weight replacement is rejected
        # while this feature is active, so duplicate storage is unnecessary.
        for name in (
            "w13_input_scale",
            "w2_input_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "g1_alphas_up",
        ):
            param = getattr(module, name, None)
            if isinstance(param, torch.nn.Parameter):
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        for source_name, payload_name in (
            ("w13_weight_scale", "w13_blockscale_swizzled"),
            ("w2_weight_scale", "w2_blockscale_swizzled"),
        ):
            source = getattr(module, source_name, None)
            derived = getattr(module, payload_name, None)
            if isinstance(source, torch.nn.Parameter) and source is not derived:
                source.data = torch.empty(0, dtype=source.dtype, device=source.device)

    def allocate_gpu_cache(self, device: torch.device) -> None:
        if not self.enabled or self._finalized:
            return
        if not self._host_attached:
            raise RuntimeError("MoE expert host payload must be attached first")
        self._allocate_gpu_cache(device)
        self._finalized = True

    def _allocate_gpu_cache(self, device: torch.device) -> None:
        # One shared compact staging area can hold the largest complete host
        # layer.  Per-key flat storage permits layers with different trailing
        # shapes while preserving their exact tensor views.
        payload_names = tuple(self.layers[0].host_payload)
        if not self._next_layer:
            self._next_layer = {
                id(current): following
                for current, following in zip(self.layers, self.layers[1:])
            }
        for layer in self.layers[1:]:
            if tuple(layer.host_payload) != payload_names:
                raise RuntimeError(
                    "all offloaded MoE layers must use the same payload layout, got "
                    f"{payload_names} and {tuple(layer.host_payload)}"
                )
        staging_templates = {
            name: max(
                (layer.host_payload[name] for layer in self.layers),
                key=lambda tensor: tensor.numel(),
            )
            for name in payload_names
        }
        staging_bytes = sum(
            tensor_nbytes(tensor) for tensor in staging_templates.values()
        )
        if staging_bytes > self.gpu_budget_bytes:
            raise RuntimeError(
                "--moe-expert-cache-gb is too small for one complete compact "
                f"staging layer: need {staging_bytes / 2**30:.3f} GiB, "
                f"budget is {self.gpu_budget_bytes / 2**30:.3f} GiB"
            )
        self.staging = {
            name: torch.empty(tensor.numel(), dtype=tensor.dtype, device=device)
            for name, tensor in staging_templates.items()
        }
        remaining = self.gpu_budget_bytes - staging_bytes
        capacities = [0] * len(self.layers)
        while True:
            progressed = False
            for i, layer in enumerate(self.layers):
                if capacities[i] >= layer.num_experts:
                    continue
                if remaining < layer.expert_bytes:
                    continue
                capacities[i] += 1
                remaining -= layer.expert_bytes
                progressed = True
            if not progressed:
                break
        for layer, capacity in zip(self.layers, capacities):
            layer.allocate(capacity, device)
        self.copy_stream = torch.cuda.Stream(device=device)
        logger.info(
            "MoE expert cache initialized: host_layers=%d host=%.3f GiB "
            "staging=%.3f GiB slab=%.3f GiB slots=%d policy=%s lease=%d "
            "drop_threshold=%.3f layer_slots=%s",
            len(self.layers),
            sum(layer.layer_bytes for layer in self.layers) / 2**30,
            staging_bytes / 2**30,
            (self.gpu_budget_bytes - staging_bytes - remaining) / 2**30,
            sum(capacities),
            self.policy,
            self.lease,
            self.drop_threshold,
            {layer.layer_id: layer.capacity for layer in self.layers},
        )

    def prepare(
        self,
        layer_cache: LayerExpertCache,
        dispatch_output,
        *,
        is_prefill: bool,
    ) -> PreparedExpertPayload:
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        topk = dispatch_output.topk_output
        resident_before = (
            ()
            if is_prefill and not self.prefill_cache_reuse
            else tuple(layer_cache.policy.expert_to_slot)
        )
        exact_routes = self.drop_threshold == 0.0
        single_token_ids_cpu = None
        if exact_routes:
            weights = topk.topk_weights
            ids = topk.topk_ids
            if ids.ndim == 2 and ids.shape[0] == 1:
                single_token_ids_cpu = ids[0].to("cpu").tolist()
                requested = list(dict.fromkeys(single_token_ids_cpu))
            else:
                requested = torch.unique(ids).to("cpu").tolist()
            dropped_count = 0
            dropped_mass = 0.0
        else:
            weights, ids, dropped = apply_miss_only_route_drop(
                topk.topk_weights,
                topk.topk_ids,
                resident_before,
                self.drop_threshold,
            )
            retained = weights > 0
            requested = torch.unique(ids[retained]).to("cpu").tolist()
            dropped_count = int(dropped.sum().item())
            dropped_mass = float(
                topk.topk_weights.masked_select(dropped).sum().item()
            )
        direct_resident = (
            exact_routes
            and not is_prefill
            and len(requested) <= layer_cache.capacity
        )
        if direct_resident:
            prediction_order = layer_cache.predicted_order
            prediction = layer_cache.predicted_experts
            prefetched = layer_cache.prefetched_experts
            requested_set = set(requested)
            for rank, expert in enumerate(prediction_order):
                while len(self._prefetch_rank_attempts) <= rank:
                    self._prefetch_rank_attempts.append(0)
                    self._prefetch_rank_hits.append(0)
                self._prefetch_rank_attempts[rank] += 1
                if expert in requested_set:
                    self._prefetch_rank_hits[rank] += 1
            stats_prefetch_predictions = len(prediction)
            stats_prefetch_matches = len(prediction & requested_set)
            useful = len(prefetched & requested_set)
            stats_prefetch_useful = useful
            stats_prefetch_wasted = len(prefetched) - useful
            layer_cache.predicted_experts = set()
            layer_cache.predicted_order = ()
            layer_cache.prefetched_experts = set()
        else:
            stats_prefetch_predictions = 0
            stats_prefetch_matches = 0
            stats_prefetch_useful = 0
            stats_prefetch_wasted = 0
        access = layer_cache.policy.access(
            requested,
            prefill=is_prefill,
            prefill_reuse=self.prefill_cache_reuse,
            allow_lease_break=direct_resident,
        )

        stats_delta = ExpertCacheStats(
            forwards=1,
            hits=len(access.hits),
            misses=len(requested) - len(access.hits),
            admissions=len(access.admissions),
            evictions=len(access.evictions),
            transient=len(access.transient),
            route_drops=dropped_count,
            dropped_mass=dropped_mass,
            prefill_reuse=len(access.hits) if is_prefill else 0,
            prefetch_predictions=stats_prefetch_predictions,
            prefetch_matches=stats_prefetch_matches,
            prefetch_useful=stats_prefetch_useful,
            prefetch_wasted=stats_prefetch_wasted,
        )

        topk = StandardTopKOutput(
            topk_weights=weights,
            topk_ids=ids,
            router_logits=topk.router_logits,
        )

        all_hit = bool(requested) and len(access.hits) == len(requested)
        main_stream = torch.cuda.current_stream()
        if all_hit:
            used_slots = []
            for expert in requested:
                slot = access.hits[expert]
                event = layer_cache.slot_events[slot]
                if event is not None:
                    main_stream.wait_event(event)
                used_slots.append(slot)
            remapped = (
                remap_single_token_expert_ids(
                    ids, single_token_ids_cpu, access.hits
                )
                if single_token_ids_cpu is not None
                else remap_expert_ids(ids, access.hits)
            )
            topk = topk._replace(topk_ids=remapped)
            prepared_dispatch = dispatch_output._replace(topk_output=topk)
            result = PreparedExpertPayload(
                prepared_dispatch,
                layer_cache.resident_payload,
                layer_cache,
                tuple(used_slots),
                False,
            )
            self._schedule_cross_layer_prefetch(layer_cache, direct_resident)
            self._commit_stats(layer_cache, stats_delta)
            return result

        if self.copy_stream is None:
            raise RuntimeError("expert cache GPU storage has not been initialized")

        if direct_resident:
            if access.transient or len(access.admissions) != stats_delta.misses:
                raise RuntimeError(
                    "direct resident execution failed to admit every miss"
                )
            expert_to_slot = {**access.hits, **access.admissions}
            for slot in dict.fromkeys(access.hits.values()):
                event = layer_cache.slot_events[slot]
                if event is not None:
                    main_stream.wait_event(event)
            with torch.cuda.stream(self.copy_stream):
                for slot in dict.fromkeys(access.admissions.values()):
                    event = layer_cache.slot_events[slot]
                    if event is not None:
                        self.copy_stream.wait_event(event)
                copies = []
                for name, host_tensor in layer_cache.host_payload.items():
                    resident = layer_cache.resident_payload[name]
                    for expert, slot in access.admissions.items():
                        copies.append((resident[slot], host_tensor[expert]))
                self._submit_copies(copies, stats_delta)
                copy_done = torch.cuda.Event()
                copy_done.record(self.copy_stream)
            main_stream.wait_event(copy_done)

            remapped = (
                remap_single_token_expert_ids(
                    ids, single_token_ids_cpu, expert_to_slot
                )
                if single_token_ids_cpu is not None
                else remap_expert_ids(ids, expert_to_slot)
            )
            topk = topk._replace(topk_ids=remapped)
            prepared_dispatch = dispatch_output._replace(topk_output=topk)
            stats_delta.h2d_bytes = stats_delta.misses * layer_cache.expert_bytes
            result = PreparedExpertPayload(
                prepared_dispatch,
                layer_cache.resident_payload,
                layer_cache,
                tuple(expert_to_slot[expert] for expert in requested),
                False,
            )
            self._schedule_cross_layer_prefetch(layer_cache, direct_resident)
            self._commit_stats(layer_cache, stats_delta)
            return result

        if self.staging_event is not None:
            self.copy_stream.wait_event(self.staging_event)

        compact_index = {expert: index for index, expert in enumerate(requested)}
        compact_payload = {}
        with torch.cuda.stream(self.copy_stream):
            for slot in dict.fromkeys(access.hits.values()):
                event = layer_cache.slot_events[slot]
                if event is not None:
                    self.copy_stream.wait_event(event)
            populate_copies = []
            for name, host_tensor in layer_cache.host_payload.items():
                expert_numel = host_tensor[0].numel()
                target = self.staging[name].narrow(
                    0, 0, len(requested) * expert_numel
                ).view(len(requested), *host_tensor.shape[1:])
                compact_payload[name] = target
                for expert in requested:
                    index = compact_index[expert]
                    if expert in access.hits:
                        slot = access.hits[expert]
                        populate_copies.append(
                            (target[index], layer_cache.resident_payload[name][slot])
                        )
                    else:
                        populate_copies.append((target[index], host_tensor[expert]))
            self._submit_copies(populate_copies, stats_delta)

            for slot in dict.fromkeys(access.admissions.values()):
                event = layer_cache.slot_events[slot]
                if event is not None:
                    self.copy_stream.wait_event(event)
            admission_copies = []
            for name in layer_cache.host_payload:
                target = compact_payload[name]
                resident = layer_cache.resident_payload[name]
                for expert, slot in access.admissions.items():
                    admission_copies.append(
                        (resident[slot], target[compact_index[expert]])
                    )
            self._submit_copies(admission_copies, stats_delta)
            copy_done = torch.cuda.Event()
            copy_done.record(self.copy_stream)
        main_stream.wait_event(copy_done)

        remapped = (
            remap_single_token_expert_ids(
                ids, single_token_ids_cpu, compact_index
            )
            if single_token_ids_cpu is not None
            else remap_expert_ids(ids, compact_index)
        )
        topk = topk._replace(topk_ids=remapped)
        prepared_dispatch = dispatch_output._replace(topk_output=topk)
        stage_bytes = sum(tensor_nbytes(tensor) for tensor in compact_payload.values())
        stats_delta.h2d_bytes = stats_delta.misses * layer_cache.expert_bytes
        stats_delta.d2d_bytes = (
            stats_delta.hits + stats_delta.admissions
        ) * layer_cache.expert_bytes
        stats_delta.staging_peak_bytes = stage_bytes
        result = PreparedExpertPayload(
            prepared_dispatch,
            compact_payload,
            layer_cache,
            tuple(access.hits.values()) + tuple(access.admissions.values()),
            True,
        )
        self._schedule_cross_layer_prefetch(layer_cache, direct_resident)
        self._commit_stats(layer_cache, stats_delta)
        return result

    def _submit_copies(
        self,
        copies: Sequence[tuple[torch.Tensor, torch.Tensor]],
        stats_delta: ExpertCacheStats,
        *,
        prefer_overlap: bool = False,
    ) -> None:
        if not copies:
            return
        if _CUDA_BATCH_COPY.submit(
            copies, self.copy_stream, prefer_overlap=prefer_overlap
        ):
            stats_delta.batched_copy_submissions += 1
            stats_delta.batched_copy_operations += len(copies)
            return
        for destination, source in copies:
            destination.copy_(source, non_blocking=True)
        stats_delta.fallback_copy_operations += len(copies)

    def _schedule_cross_layer_prefetch(
        self, layer_cache: LayerExpertCache, eligible: bool
    ) -> None:
        predicted = self._pending_prefetch.pop(id(layer_cache), ())
        if not eligible or not self.cross_layer_prefetch or not predicted:
            return
        next_layer = self._next_layer.get(id(layer_cache))
        if next_layer is None:
            return
        prefetch_count = self.recommended_prefetch_count(len(predicted))
        selected = predicted[:prefetch_count]
        if len(selected) > next_layer.capacity:
            return
        if self.copy_stream is None:
            raise RuntimeError("expert cache GPU storage has not been initialized")

        access = next_layer.policy.access(selected, allow_lease_break=True)
        if access.transient:
            raise RuntimeError(
                "cross-layer prefetch failed to admit its bounded expert set"
            )
        next_layer.predicted_experts = set(predicted)
        next_layer.predicted_order = predicted
        next_layer.prefetched_experts = set(access.admissions)
        delta = ExpertCacheStats(
            prefetch_h2d_bytes=len(access.admissions) * next_layer.expert_bytes,
            prefetch_selected=len(selected),
            prefetch_admissions=len(access.admissions),
            prefetch_evictions=len(access.evictions),
        )
        if not access.admissions:
            next_layer.stats.add(delta)
            self.stats.add(delta)
            return

        with torch.cuda.stream(self.copy_stream):
            for slot in dict.fromkeys(access.admissions.values()):
                event = next_layer.slot_events[slot]
                if event is not None:
                    self.copy_stream.wait_event(event)
            copies = []
            for name, host_tensor in next_layer.host_payload.items():
                resident = next_layer.resident_payload[name]
                for expert, slot in access.admissions.items():
                    copies.append((resident[slot], host_tensor[expert]))
            self._submit_copies(copies, delta, prefer_overlap=True)
            prefetch_done = torch.cuda.Event()
            prefetch_done.record(self.copy_stream)
        for slot in access.admissions.values():
            next_layer.slot_events[slot] = prefetch_done
        next_layer.stats.add(delta)
        self.stats.add(delta)

    def recommended_prefetch_count(self, predicted_count: int) -> int:
        if predicted_count <= 0:
            return 0
        minimum = max(1, math.ceil(predicted_count * self.prefetch_min_ratio))
        count = predicted_count
        while count > minimum:
            rank = count - 1
            if (
                rank >= len(self._prefetch_rank_attempts)
                or self._prefetch_rank_attempts[rank] < _PREFETCH_MIN_SAMPLES
            ):
                break
            hit_rate = (
                self._prefetch_rank_hits[rank]
                / self._prefetch_rank_attempts[rank]
            )
            if hit_rate >= _PREFETCH_MIN_HIT_RATE:
                break
            count -= 1
        return count

    @eager_on_graph(True)
    def predict_next_layer(
        self, layer_cache: LayerExpertCache, hidden_states: torch.Tensor
    ) -> None:
        if (
            not self.cross_layer_prefetch
            or self.drop_threshold != 0.0
            or hidden_states.ndim != 2
            or hidden_states.shape[0] == 0
            or layer_cache.capacity == 0
        ):
            return
        next_router = self._next_router.get(id(layer_cache))
        next_layer = self._next_layer.get(id(layer_cache))
        if next_router is None or next_layer is None:
            return

        router_logits, _ = next_router.gate(hidden_states)
        topk_output = next_router.topk(hidden_states, router_logits)
        predicted_cpu = topk_output.topk_ids.flatten().to("cpu").tolist()
        predicted = tuple(dict.fromkeys(int(expert) for expert in predicted_cpu))
        if len(predicted) <= next_layer.capacity:
            self._pending_prefetch[id(layer_cache)] = predicted

    def _commit_stats(
        self, layer: LayerExpertCache, stats_delta: ExpertCacheStats
    ) -> None:
        layer.stats.add(stats_delta)
        self.stats.add(stats_delta)
        if self.stats.forwards % 1000 == 0:
            logger.info(
                "MoE expert cache stats: forwards=%d hit=%d miss=%d "
                "h2d=%.3f GiB d2d=%.3f GiB prefetch_h2d=%.3f GiB "
                "copy_batch=%d/%d copy_fallback=%d "
                "prefetch_selected=%d prefetch_admit=%d "
                "prefetch_useful=%d prefetch_wasted=%d "
                "prefetch_match=%d/%d "
                "admit=%d evict=%d transient=%d "
                "route_drop=%d dropped_mass=%.6f prefill_reuse=%d stage_peak=%.3f MiB",
                self.stats.forwards,
                self.stats.hits,
                self.stats.misses,
                self.stats.h2d_bytes / 2**30,
                self.stats.d2d_bytes / 2**30,
                self.stats.prefetch_h2d_bytes / 2**30,
                self.stats.batched_copy_submissions,
                self.stats.batched_copy_operations,
                self.stats.fallback_copy_operations,
                self.stats.prefetch_selected,
                self.stats.prefetch_admissions,
                self.stats.prefetch_useful,
                self.stats.prefetch_wasted,
                self.stats.prefetch_matches,
                self.stats.prefetch_predictions,
                self.stats.admissions,
                self.stats.evictions,
                self.stats.transient,
                self.stats.route_drops,
                self.stats.dropped_mass,
                self.stats.prefill_reuse,
                self.stats.staging_peak_bytes / 2**20,
            )
            if self.stats.forwards % 10000 == 0:
                logger.info(
                    "MoE expert cache per-layer stats: %s",
                    {
                        item.layer_id: {
                            "hit": item.stats.hits,
                            "miss": item.stats.misses,
                            "h2d_mib": round(item.stats.h2d_bytes / 2**20, 3),
                            "d2d_mib": round(item.stats.d2d_bytes / 2**20, 3),
                            "admit": item.stats.admissions,
                            "evict": item.stats.evictions,
                            "transient": item.stats.transient,
                            "drop": item.stats.route_drops,
                            "drop_mass": round(item.stats.dropped_mass, 6),
                            "prefill_reuse": item.stats.prefill_reuse,
                            "prefetch_h2d_mib": round(
                                item.stats.prefetch_h2d_bytes / 2**20, 3
                            ),
                            "prefetch_admit": item.stats.prefetch_admissions,
                            "prefetch_selected": item.stats.prefetch_selected,
                            "prefetch_match": item.stats.prefetch_matches,
                            "prefetch_predictions": item.stats.prefetch_predictions,
                            "prefetch_useful": item.stats.prefetch_useful,
                            "prefetch_wasted": item.stats.prefetch_wasted,
                        }
                        for item in self.layers
                    },
                )


_MANAGER: Optional[MoeExpertCacheManager] = None


def get_moe_expert_cache_manager() -> MoeExpertCacheManager:
    global _MANAGER
    if _MANAGER is None:
        from sglang.srt.runtime_context import get_server_args

        args = get_server_args()
        _MANAGER = MoeExpertCacheManager(
            host_budget_bytes=int(args.moe_expert_offload_gb * 2**30),
            gpu_budget_bytes=int(args.moe_expert_cache_gb * 2**30),
            policy=args.moe_expert_cache_policy,
            lease=args.moe_expert_cache_lease,
            drop_threshold=args.moe_expert_drop_threshold,
            prefill_cache_reuse=args.moe_prefill_cache_reuse,
            cross_layer_prefetch=args.moe_expert_cache_cross_layer_prefetch,
            prefetch_min_ratio=args.moe_expert_cache_prefetch_min_ratio,
            resident_shallow_layers=args.moe_expert_resident_shallow_layers,
        )
    return _MANAGER


def reset_moe_expert_cache_manager() -> None:
    global _MANAGER
    _MANAGER = None


def initialize_moe_expert_cache(model: torch.nn.Module, device: torch.device) -> None:
    manager = get_moe_expert_cache_manager()
    if manager.enabled:
        manager.attach_model(model, device)


def allocate_moe_expert_gpu_cache(device: torch.device) -> None:
    manager = get_moe_expert_cache_manager()
    if manager.enabled:
        manager.allocate_gpu_cache(device)
