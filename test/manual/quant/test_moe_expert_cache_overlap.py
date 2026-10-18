"""Run only through profile_output/ram_guard.py with MAX_JOBS=1."""

import pytest
import torch

from sglang.srt.layers.moe.expert_cache import LayerExpertCache, MoeExpertCacheManager
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    fused_experts_none_to_triton,
    run_expert_cache_overlap,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def make_payload(experts=6, hidden=128, intermediate=128, dtype=torch.bfloat16):
    torch.manual_seed(41)
    payload = {}
    for name, rows, cols in (
        ("w13", 2 * intermediate, hidden),
        ("w2", hidden, intermediate),
    ):
        payload[name + "_qweight"] = torch.randint(
            0, 256, (experts, rows, cols // 2), dtype=torch.uint8
        ).pin_memory()
        payload[name + "_scales"] = (
            torch.rand((experts, rows, cols // 128), dtype=dtype) * 0.01 + 0.001
        ).pin_memory()
    return payload


def quant_info(payload):
    return TritonMoeQuantInfo(
        w13_weight=payload["w13_qweight"],
        w2_weight=payload["w2_qweight"],
        w13_scale=payload["w13_scales"],
        w2_scale=payload["w2_scales"],
        use_int4_w4a16=True,
        block_shape=[0, 128],
    )


def reference_moe(dispatch, payload, factor):
    """Independent FP32 dequantization and per-lane expert computation."""

    def unpack(prefix):
        packed = payload[prefix + "_qweight"].to("cuda")
        values = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        scales = payload[prefix + "_scales"].to("cuda").float()
        return (values.float() - 8) * scales.repeat_interleave(128, dim=-1)

    w13, w2 = unpack("w13"), unpack("w2")
    output = torch.zeros_like(dispatch.hidden_states, dtype=torch.float32)
    for token, routes in enumerate(dispatch.topk_output.topk_ids.cpu().tolist()):
        for lane, expert in enumerate(routes):
            gate, up = (w13[expert] @ dispatch.hidden_states[token].float()).chunk(2)
            value = w2[expert] @ (torch.nn.functional.silu(gate) * up)
            output[token] += value * dispatch.topk_output.topk_weights[token, lane]
    return output * factor


def make_manager(payload, capacity, execution="overlap", layers=1):
    expert_bytes = sum(t[0].numel() * t.element_size() for t in payload.values())
    layer_bytes = expert_bytes * payload["w13_qweight"].shape[0]
    manager = MoeExpertCacheManager(
        host_budget_bytes=layers * layer_bytes,
        gpu_budget_bytes=layer_bytes + layers * capacity * expert_bytes,
        policy="lru",
        lease=0,
        drop_threshold=0,
        prefill_cache_reuse=True,
        cross_layer_prefetch=False,
        execution=execution,
    )
    manager.layers = [
        LayerExpertCache(
            layer_id=i,
            host_payload=payload,
            num_experts=payload["w13_qweight"].shape[0],
            policy_name="lru",
            lease=0,
            manager=manager,
            payload_format="moe_wna16",
        )
        for i in range(layers)
    ]
    manager._allocate_gpu_cache(torch.device("cuda"))
    return manager


def make_dispatch(ids, hidden=128, dtype=torch.bfloat16):
    ids = torch.tensor(ids, dtype=torch.int64, device="cuda")
    # Neither normalized nor uniform: detect omitted/double route weighting.
    weights = torch.arange(1, ids.numel() + 1, dtype=torch.float32, device="cuda")
    weights = weights.reshape_as(ids) / (ids.numel() + 2)
    if ids.shape[0] > 1:
        weights[0, -1] = 0
    return StandardDispatchOutput(
        hidden_states=torch.randn((ids.shape[0], hidden), device="cuda", dtype=dtype),
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(weights, ids, torch.empty(0, device="cuda")),
    )


def run_prepared(prepared, config):
    if prepared.overlap is not None:
        result = run_expert_cache_overlap(prepared, quant_info, config)
    else:
        result = fused_experts_none_to_triton(
            prepared.dispatch_output, quant_info(prepared.payload), config
        )
    prepared.mark_kernel_launched()
    return result.hidden_states


@pytest.mark.parametrize("batch_size", (1, 4, 8))
@pytest.mark.parametrize("capacity", (0, 1, 6))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_overlap_matches_serial_and_preserves_inputs(batch_size, capacity, dtype):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    payload = make_payload(dtype=dtype)
    manager = make_manager(payload, capacity)
    layer = manager.layers[0]
    config = MoeRunnerConfig(
        num_experts=6, num_local_experts=6, top_k=2, routed_scaling_factor=1.7
    )
    # Seed the cache with an actual kernel, not just policy metadata.
    seed = manager.prepare(
        layer, make_dispatch([[0, 1]], dtype=dtype), is_prefill=False
    )
    run_prepared(seed, config)
    d2d_before = layer.stats.d2d_bytes

    dispatch = make_dispatch([[0, 2], [3, 0]] * 4, dtype=dtype)
    dispatch = dispatch._replace(
        hidden_states=dispatch.hidden_states[:batch_size],
        topk_output=dispatch.topk_output._replace(
            topk_ids=dispatch.topk_output.topk_ids[:batch_size],
            topk_weights=dispatch.topk_output.topk_weights[:batch_size],
        ),
    )
    original_input = dispatch.hidden_states.clone()
    original_weights = dispatch.topk_output.topk_weights.clone()
    gpu_payload = {name: tensor.to("cuda") for name, tensor in payload.items()}
    reference_dispatch = dispatch._replace(hidden_states=original_input.clone())
    expected = fused_experts_none_to_triton(
        reference_dispatch, quant_info(gpu_payload), config
    ).hidden_states
    prepared = manager.prepare(
        layer, dispatch, is_prefill=False, is_decode=True, allow_overlap=True
    )
    assert (prepared.overlap is not None) == (capacity > 0)
    actual = run_prepared(prepared, config)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)
    torch.testing.assert_close(
        actual.float(),
        reference_moe(dispatch._replace(hidden_states=original_input), payload, 1.7),
        atol=2e-2,
        rtol=0,
    )
    torch.testing.assert_close(
        dispatch.topk_output.topk_weights, original_weights, atol=0, rtol=0
    )
    if prepared.overlap is not None:
        torch.testing.assert_close(
            dispatch.hidden_states, original_input, atol=0, rtol=0
        )
        assert prepared.uses_staging == (capacity == 1)
        assert layer.stats.d2d_bytes == d2d_before  # no hit-to-staging copies
    for expert, slot in layer.policy.expert_to_slot.items():
        for name, host_tensor in payload.items():
            torch.testing.assert_close(
                layer.resident_payload[name][slot].cpu(),
                host_tensor[expert],
                atol=0,
                rtol=0,
            )


def test_overlap_reuses_staging_and_handles_eviction_and_prefetch():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    payload = make_payload()
    manager = make_manager(payload, capacity=2, layers=2)
    config = MoeRunnerConfig(num_experts=6, num_local_experts=6, top_k=2)
    gpu_payload = {name: tensor.to("cuda") for name, tensor in payload.items()}
    saw_staging_admission = False
    saw_full_hit = False
    for routes in ([[0, 1]], [[0, 2], [3, 0]], [[0, 2]], [[4, 5]], [[4, 5]]):
        for layer in manager.layers:
            dispatch = make_dispatch(routes)
            reference = dispatch._replace(hidden_states=dispatch.hidden_states.clone())
            expected = fused_experts_none_to_triton(
                reference, quant_info(gpu_payload), config
            ).hidden_states
            prepared = manager.prepare(
                layer, dispatch, is_prefill=False, is_decode=True, allow_overlap=True
            )
            saw_staging_admission |= (
                prepared.overlap is not None and prepared.uses_staging
            )
            saw_full_hit |= (
                prepared.overlap is None
                and not prepared.uses_staging
                and layer.stats.hits > 0
            )
            actual = run_prepared(prepared, config)
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)
    assert saw_staging_admission and saw_full_hit
    assert manager.stats.evictions > 0

    manager.cross_layer_prefetch = True
    manager._pending_prefetch[id(manager.layers[0])] = (1, 2)
    manager._schedule_cross_layer_prefetch(manager.layers[0], True)
    # No synchronize: consuming a prefetched hit must respect its copy event.
    dispatch = make_dispatch([[1, 3]])
    reference = dispatch._replace(hidden_states=dispatch.hidden_states.clone())
    expected = fused_experts_none_to_triton(
        reference, quant_info(gpu_payload), config
    ).hidden_states
    prepared = manager.prepare(
        manager.layers[1],
        dispatch,
        is_prefill=False,
        is_decode=True,
        allow_overlap=True,
    )
    assert prepared.overlap is not None
    torch.testing.assert_close(
        run_prepared(prepared, config), expected, atol=2e-2, rtol=0
    )


@pytest.mark.parametrize("top_k", (1, 3, 8))
def test_overlap_other_topk_widths_and_duplicate_lanes(top_k):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    payload = make_payload()
    manager = make_manager(payload, capacity=2)
    layer = manager.layers[0]
    config = MoeRunnerConfig(num_experts=6, num_local_experts=6, top_k=top_k)
    seed = manager.prepare(layer, make_dispatch([[0, 1]]), is_prefill=False)
    run_prepared(seed, config)
    rows = [[(i + j) % 4 for j in range(top_k)] for i in range(4)]
    dispatch = make_dispatch(rows)
    original_input = dispatch.hidden_states.clone()
    prepared = manager.prepare(
        layer, dispatch, is_prefill=False, is_decode=True, allow_overlap=True
    )
    assert prepared.overlap is not None
    actual = run_prepared(prepared, config)
    expected = reference_moe(
        dispatch._replace(hidden_states=original_input), payload, 1.0
    )
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=0)


@pytest.mark.parametrize(
    "reason", ("prefill", "not_decode", "other_format", "disabled")
)
def test_overlap_eligibility_preserves_existing_paths(reason):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    payload = make_payload()
    manager = make_manager(payload, capacity=2)
    layer = manager.layers[0]
    config = MoeRunnerConfig(num_experts=6, num_local_experts=6, top_k=2)
    run_prepared(
        manager.prepare(layer, make_dispatch([[0, 1]]), is_prefill=False), config
    )
    before = layer.policy.step
    if reason == "other_format":
        layer.payload_format = "unknown"
    prepared = manager.prepare(
        layer,
        make_dispatch([[0, 2]]),
        is_prefill=reason == "prefill",
        is_decode=reason not in ("prefill", "not_decode"),
        allow_overlap=reason != "disabled",
    )
    assert prepared.overlap is None
    if reason == "prefill":
        assert layer.policy.step == before
        assert 2 not in layer.policy.expert_to_slot
    run_prepared(prepared, config)
