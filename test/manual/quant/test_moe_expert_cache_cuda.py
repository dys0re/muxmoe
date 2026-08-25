import pytest
import torch

from sglang.srt.layers.moe.expert_cache import (
    ExpertCacheStats,
    LayerExpertCache,
    MoeExpertCacheManager,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput


def _host_payload(num_experts: int = 3):
    shapes = {
        "w13_weight": (num_experts, 8, 4),
        "w2_weight": (num_experts, 4, 4),
        "w13_blockscale_swizzled": (num_experts, 8, 4),
        "w2_blockscale_swizzled": (num_experts, 4, 4),
        "g1_alphas": (num_experts,),
        "g2_alphas": (num_experts,),
    }
    dtypes = {
        "w13_weight": torch.uint8,
        "w2_weight": torch.uint8,
        "w13_blockscale_swizzled": torch.float8_e4m3fn,
        "w2_blockscale_swizzled": torch.float8_e4m3fn,
        "g1_alphas": torch.float32,
        "g2_alphas": torch.float32,
    }
    payload = {}
    for name, shape in shapes.items():
        tensor = torch.empty(shape, dtype=dtypes[name], pin_memory=True)
        for expert in range(num_experts):
            tensor[expert].fill_(expert + 1)
        payload[name] = tensor
    return payload


def _dispatch(experts):
    ids = torch.tensor([experts], dtype=torch.int64, device="cuda")
    weights = torch.ones_like(ids, dtype=torch.float32)
    weights /= weights.shape[1]
    topk = StandardTopKOutput(
        topk_weights=weights,
        topk_ids=ids,
        router_logits=torch.empty(0, device="cuda"),
    )
    return StandardDispatchOutput(
        hidden_states=torch.empty((1, 4), dtype=torch.bfloat16, device="cuda"),
        hidden_states_scale=None,
        topk_output=topk,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_staging_full_hit_partial_miss_and_eviction():
    host_payload = _host_payload()
    expert_bytes = sum(
        tensor[0].numel() * tensor.element_size()
        for tensor in host_payload.values()
    )
    layer_bytes = expert_bytes * 3
    manager = MoeExpertCacheManager(
        host_budget_bytes=layer_bytes,
        gpu_budget_bytes=layer_bytes + 2 * expert_bytes,
        policy="lru",
        lease=0,
        drop_threshold=0.1,
        prefill_cache_reuse=True,
    )
    layer = LayerExpertCache(
        layer_id=0,
        host_payload=host_payload,
        num_experts=3,
        policy_name="lru",
        lease=0,
        manager=manager,
        stats=ExpertCacheStats(),
    )
    manager.layers = [layer]
    manager._allocate_gpu_cache(torch.device("cuda"))
    assert layer.capacity == 2

    first = manager.prepare(layer, _dispatch([0, 1]), is_prefill=False)
    assert first.uses_staging is True
    first.mark_kernel_launched()
    torch.cuda.synchronize()
    assert first.payload["w13_weight"][:, 0, 0].tolist() == [1, 2]
    assert layer.policy.expert_to_slot == {0: 0, 1: 1}

    full_hit = manager.prepare(layer, _dispatch([1, 0]), is_prefill=False)
    assert full_hit.uses_staging is False
    assert full_hit.dispatch_output.topk_output.topk_ids.tolist() == [[1, 0]]
    full_hit.mark_kernel_launched()

    partial = manager.prepare(layer, _dispatch([1, 2]), is_prefill=False)
    assert partial.uses_staging is True
    partial.mark_kernel_launched()
    torch.cuda.synchronize()
    assert partial.payload["w13_weight"][:, 0, 0].tolist() == [2, 3]
    assert layer.policy.expert_to_slot == {1: 1, 2: 0}
    assert layer.stats.h2d_bytes > 0
    assert layer.stats.d2d_bytes > 0
    assert layer.stats.evictions == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_cross_layer_prefetch_populates_next_resident_slab():
    host_payloads = [_host_payload(), _host_payload()]
    expert_bytes = sum(
        tensor[0].numel() * tensor.element_size()
        for tensor in host_payloads[0].values()
    )
    layer_bytes = expert_bytes * 3
    manager = MoeExpertCacheManager(
        host_budget_bytes=2 * layer_bytes,
        gpu_budget_bytes=layer_bytes + 4 * expert_bytes,
        policy="lru",
        lease=0,
        drop_threshold=0,
        prefill_cache_reuse=True,
        cross_layer_prefetch=True,
    )
    layers = [
        LayerExpertCache(
            layer_id=layer_id,
            host_payload=host_payload,
            num_experts=3,
            policy_name="lru",
            lease=0,
            manager=manager,
        )
        for layer_id, host_payload in enumerate(host_payloads)
    ]
    manager.layers = layers
    manager._allocate_gpu_cache(torch.device("cuda"))
    assert [layer.capacity for layer in layers] == [2, 2]

    manager._pending_prefetch[id(layers[0])] = (1, 2)
    current = manager.prepare(layers[0], _dispatch([0, 1]), is_prefill=False)
    current.mark_kernel_launched()
    torch.cuda.synchronize()

    assert layers[1].policy.expert_to_slot == {1: 0, 2: 1}
    assert layers[1].prefetched_experts == {1, 2}
    assert layers[1].stats.prefetch_admissions == 2
    assert layers[1].stats.prefetch_h2d_bytes == 2 * expert_bytes
    assert layers[1].predicted_experts == {1, 2}
    for expert, slot in layers[1].policy.expert_to_slot.items():
        assert layers[1].resident_payload["w13_weight"][slot, 0, 0] == expert + 1

    following = manager.prepare(layers[1], _dispatch([1, 2]), is_prefill=False)
    following.mark_kernel_launched()
    torch.cuda.synchronize()
    assert following.uses_staging is False
    assert layers[1].stats.h2d_bytes == 0
    assert layers[1].stats.prefetch_predictions == 2
    assert layers[1].stats.prefetch_matches == 2
    assert layers[1].stats.prefetch_useful == 2
    assert layers[1].stats.prefetch_wasted == 0
