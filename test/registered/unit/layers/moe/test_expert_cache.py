import json
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.moe.expert_cache as expert_cache_module
from sglang.srt.layers.moe.expert_cache import (
    ExpertCachePolicy,
    ExpertCacheStats,
    ExpertPayloadSpec,
    LayerExpertCache,
    MoeExpertCacheManager,
    apply_miss_only_route_drop,
    modelopt_nvfp4_payload_specs,
    moe_wna16_payload_specs,
    remap_expert_ids,
    remap_single_token_expert_ids,
    select_complete_layers,
)


def test_payload_budget_uses_complete_layers_and_final_padded_shapes():
    specs = modelopt_nvfp4_payload_specs(
        num_experts=4,
        hidden_size=96,
        intermediate_size=64,
        group_size=16,
        is_gated=True,
    )
    by_name = {spec.name: spec for spec in specs}
    assert by_name["w13_weight"].shape == (4, 128, 48)
    assert by_name["w2_weight"].shape == (4, 96, 32)
    assert by_name["w13_blockscale_swizzled"].shape == (4, 128, 8)
    assert by_name["w2_blockscale_swizzled"].shape == (4, 128, 4)

    layer_bytes = sum(spec.nbytes for spec in specs)
    assert select_complete_layers(
        [layer_bytes, layer_bytes, layer_bytes], 2 * layer_bytes - 1
    ) == [0]
    assert select_complete_layers(
        [layer_bytes, layer_bytes, layer_bytes], 2 * layer_bytes
    ) == [0, 1]


def test_qwen35b_metadata_selects_37_complete_host_layers():
    specs = modelopt_nvfp4_payload_specs(
        num_experts=256,
        hidden_size=2048,
        intermediate_size=512,
        group_size=16,
        is_gated=True,
    )
    layer_bytes = sum(spec.nbytes for spec in specs)
    selected = select_complete_layers([layer_bytes] * 40, 16 * 2**30)
    assert len(selected) == 37
    assert layer_bytes == 432 * 2**20 + 2 * 2**10
    assert len(selected) * layer_bytes <= 16 * 2**30
    assert (len(selected) + 1) * layer_bytes > 16 * 2**30


def test_qwen36_gptq_payload_uses_complete_runtime_tensors():
    specs = moe_wna16_payload_specs(
        num_experts=256,
        hidden_size=2048,
        intermediate_size=512,
        weight_bits=4,
        group_size=128,
        has_zp=False,
        dtype=torch.bfloat16,
    )
    by_name = {spec.name: spec for spec in specs}
    assert tuple(by_name) == (
        "w13_qweight",
        "w2_qweight",
        "w13_scales",
        "w2_scales",
    )
    assert by_name["w13_qweight"].shape == (256, 1024, 1024)
    assert by_name["w2_qweight"].shape == (256, 2048, 256)
    assert by_name["w13_scales"].shape == (256, 1024, 16)
    assert by_name["w2_scales"].shape == (256, 2048, 4)
    assert sum(spec.nbytes for spec in specs) == 396 * 2**20


def test_qwen36_gptq_manager_reserves_38_layers_with_headroom():
    specs = moe_wna16_payload_specs(
        num_experts=256,
        hidden_size=2048,
        intermediate_size=512,
        weight_bits=4,
        group_size=128,
        has_zp=False,
        dtype=torch.bfloat16,
    )
    manager = MoeExpertCacheManager(
        host_budget_bytes=16 * 2**30,
        gpu_budget_bytes=1 * 2**30,
        policy="lease",
        lease=3,
        drop_threshold=0,
        prefill_cache_reuse=True,
    )
    reserved = [
        manager.reserve_layer(
            torch.nn.Module(), specs, payload_format="moe_wna16"
        )
        for _ in range(40)
    ]
    assert sum(reserved) == 38
    assert manager.reservation_headroom_bytes == 3 * 396 * 2**20
    assert (
        manager.reserved_host_bytes + manager.reservation_headroom_bytes
        <= manager.host_budget_bytes
    )


def test_manager_retains_complete_layers_as_load_and_runtime_headroom():
    specs = modelopt_nvfp4_payload_specs(
        num_experts=256,
        hidden_size=2048,
        intermediate_size=512,
        group_size=16,
        is_gated=True,
    )
    manager = MoeExpertCacheManager(
        host_budget_bytes=16 * 2**30,
        gpu_budget_bytes=1 * 2**30,
        policy="lease",
        lease=3,
        drop_threshold=0,
        prefill_cache_reuse=True,
    )
    reserved = [manager.reserve_layer(torch.nn.Module(), specs) for _ in range(40)]
    assert sum(reserved) == 34
    assert manager.reservation_headroom_bytes == 3 * (432 * 2**20 + 2 * 2**10)
    assert (
        manager.reserved_host_bytes + manager.reservation_headroom_bytes
        <= manager.host_budget_bytes
    )


def test_reservation_rejects_cache_smaller_than_complete_staging_layer():
    specs = modelopt_nvfp4_payload_specs(
        num_experts=4,
        hidden_size=96,
        intermediate_size=64,
        group_size=16,
        is_gated=True,
    )
    payload_bytes = sum(spec.nbytes for spec in specs)
    manager = MoeExpertCacheManager(
        host_budget_bytes=payload_bytes,
        gpu_budget_bytes=payload_bytes - 1,
        policy="lease",
        lease=3,
        drop_threshold=0,
        prefill_cache_reuse=True,
    )
    with pytest.raises(ValueError, match="complete compact staging layer"):
        manager.reserve_layer(torch.nn.Module(), specs)


def test_lease_protects_hits_and_uses_transient_when_no_slot_is_eligible():
    policy = ExpertCachePolicy(capacity=2, policy="lease", lease=3)
    first = policy.access([0, 1])
    assert first.admissions == {0: 0, 1: 1}

    hit = policy.access([0])
    assert hit.hits == {0: 0}
    blocked = policy.access([2])
    assert blocked.transient == (2,)
    assert blocked.evictions == ()

    admitted = policy.access([2])
    assert admitted.admissions == {2: 1}
    assert admitted.evictions == (1,)
    assert policy.expert_to_slot == {0: 0, 2: 1}


def test_direct_resident_path_can_break_lease_without_evicting_hits():
    policy = ExpertCachePolicy(capacity=3, policy="lease", lease=10)
    policy.access([0, 1, 2])

    blocked = policy.access([0, 3])
    assert blocked.hits == {0: 0}
    assert blocked.transient == (3,)

    admitted = policy.access([0, 3], allow_lease_break=True)
    assert admitted.hits == {0: 0}
    assert admitted.admissions == {3: 1}
    assert admitted.evictions == (1,)
    assert policy.expert_to_slot == {0: 0, 2: 2, 3: 1}


def test_current_forward_hits_are_never_evicted():
    policy = ExpertCachePolicy(capacity=2, policy="lru", lease=0)
    policy.access([0, 1])
    access = policy.access([0, 2])
    assert access.hits == {0: 0}
    assert access.admissions == {2: 1}
    assert access.evictions == (1,)


def test_prefill_reuses_without_admission_or_lease_renewal():
    policy = ExpertCachePolicy(capacity=1, policy="lease", lease=3)
    policy.access([7])
    lease_before = policy.slots[0].lease_until
    step_before = policy.step

    access = policy.access([7, 8], prefill=True, prefill_reuse=True)
    assert access.hits == {7: 0}
    assert access.transient == (8,)
    assert policy.step == step_before
    assert policy.slots[0].lease_until == lease_before

    no_reuse = policy.access([7], prefill=True, prefill_reuse=False)
    assert no_reuse.hits == {}
    assert no_reuse.transient == (7,)


def test_miss_only_drop_preserves_hits_and_top1_then_renormalizes():
    weights = torch.tensor([[0.60, 0.09, 0.31], [0.01, 0.49, 0.50]])
    ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    kept_weights, kept_ids, dropped = apply_miss_only_route_drop(
        weights, ids, resident_experts={1, 5}, threshold=0.10
    )

    assert dropped.tolist() == [[False, False, False], [True, False, False]]
    assert kept_ids.tolist() == [[0, 1, 2], [5, 4, 5]]
    assert kept_weights.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0])
    assert kept_weights[1, 0].item() == 0

    # Even when every route is a low-weight miss, the per-token maximum stays.
    low_weights = torch.tensor([[0.04, 0.03]])
    low_ids = torch.tensor([[9, 10]])
    kept, _, low_dropped = apply_miss_only_route_drop(
        low_weights, low_ids, resident_experts=(), threshold=0.10
    )
    assert low_dropped.tolist() == [[False, True]]
    torch.testing.assert_close(kept, torch.tensor([[1.0, 0.0]]))


def test_compact_id_remapping_is_total_and_explicit():
    ids = torch.tensor([[8, 3, 8], [5, 3, 5]])
    assert remap_expert_ids(ids, {8: 0, 3: 1, 5: 2}).tolist() == [
        [0, 1, 0],
        [2, 1, 2],
    ]
    with pytest.raises(ValueError, match="missing compact IDs"):
        remap_expert_ids(ids, {8: 0, 3: 1})


def test_single_token_remapping_reuses_cpu_expert_ids():
    ids = torch.tensor([[8, 3, 8, 5]])
    remapped = remap_single_token_expert_ids(
        ids, [8, 3, 8, 5], {8: 0, 3: 1, 5: 2}
    )
    assert remapped.tolist() == [[0, 1, 0, 2]]
    with pytest.raises(ValueError, match="missing compact ID for expert 5"):
        remap_single_token_expert_ids(ids, [8, 3, 8, 5], {8: 0, 3: 1})


def test_cross_layer_router_prediction_queues_unique_experts():
    manager = MoeExpertCacheManager(
        host_budget_bytes=1,
        gpu_budget_bytes=1,
        policy="lease",
        lease=3,
        drop_threshold=0,
        prefill_cache_reuse=True,
        cross_layer_prefetch=True,
    )
    layers = [
        LayerExpertCache(
            layer_id=layer_id,
            host_payload={"weight": torch.empty((4, 1))},
            num_experts=4,
            policy_name="lease",
            lease=3,
            manager=manager,
            capacity=3,
        )
        for layer_id in range(2)
    ]

    class FakeRouter:
        @staticmethod
        def gate(hidden_states):
            return hidden_states.new_zeros((1, 4)), None

        @staticmethod
        def topk(hidden_states, router_logits):
            del hidden_states, router_logits
            return SimpleNamespace(topk_ids=torch.tensor([[2, 1, 2]]))

    manager._next_layer[id(layers[0])] = layers[1]
    manager._next_router[id(layers[0])] = FakeRouter()
    manager.predict_next_layer(layers[0], torch.zeros((1, 2)))
    assert manager._pending_prefetch[id(layers[0])] == (2, 1)


def test_cross_layer_router_prediction_flattens_batch_experts():
    manager = MoeExpertCacheManager(
        host_budget_bytes=1,
        gpu_budget_bytes=1,
        policy="lease",
        lease=3,
        drop_threshold=0,
        prefill_cache_reuse=True,
        cross_layer_prefetch=True,
    )
    layers = [
        LayerExpertCache(
            layer_id=layer_id,
            host_payload={"weight": torch.empty((8, 1))},
            num_experts=8,
            policy_name="lease",
            lease=3,
            manager=manager,
            capacity=8,
        )
        for layer_id in range(2)
    ]

    class FakeRouter:
        @staticmethod
        def gate(hidden_states):
            return hidden_states.new_zeros((2, 8)), None

        @staticmethod
        def topk(hidden_states, router_logits):
            del hidden_states, router_logits
            return SimpleNamespace(
                topk_ids=torch.tensor([[5, 2, 5], [7, 2, 3]])
            )

    manager._next_layer[id(layers[0])] = layers[1]
    manager._next_router[id(layers[0])] = FakeRouter()
    manager.predict_next_layer(layers[0], torch.zeros((2, 4)))

    assert manager._pending_prefetch[id(layers[0])] == (5, 2, 7, 3)


@pytest.mark.parametrize(
    ("minimum_ratio", "expected_count"),
    ((0.5, 4), (0.75, 6), (1.0, 8)),
)
def test_prefetch_count_respects_configured_minimum_ratio(
    minimum_ratio, expected_count
):
    manager = MoeExpertCacheManager(
        host_budget_bytes=1,
        gpu_budget_bytes=1,
        policy="lease",
        lease=3,
        drop_threshold=0,
        prefill_cache_reuse=True,
        prefetch_min_ratio=minimum_ratio,
    )
    manager._prefetch_rank_attempts = [128] * 8
    manager._prefetch_rank_hits = [128, 120, 115, 110, 100, 90, 80, 70]

    assert manager.recommended_prefetch_count(8) == expected_count
    manager._prefetch_rank_hits[7] = 110
    assert manager.recommended_prefetch_count(8) == 8


def test_prefetch_count_defaults_to_adaptive_minimum():
    manager = MoeExpertCacheManager(
        host_budget_bytes=1,
        gpu_budget_bytes=1,
        policy="lease",
        lease=3,
        drop_threshold=0,
        prefill_cache_reuse=True,
    )
    manager._prefetch_rank_attempts = [128] * 8
    manager._prefetch_rank_hits = [128, 120, 115, 110, 100, 90, 80, 70]

    assert manager.recommended_prefetch_count(8) == 4


@pytest.mark.parametrize("minimum_ratio", (0.0, -0.5, 1.01))
def test_prefetch_minimum_ratio_must_be_bounded(minimum_ratio):
    with pytest.raises(ValueError, match="prefetch_min_ratio must be in"):
        MoeExpertCacheManager(
            host_budget_bytes=1,
            gpu_budget_bytes=1,
            policy="lease",
            lease=3,
            drop_threshold=0,
            prefill_cache_reuse=True,
            prefetch_min_ratio=minimum_ratio,
        )


def test_reserve_layer_keeps_configured_shallow_layers_device_resident():
    manager = MoeExpertCacheManager(
        host_budget_bytes=128,
        gpu_budget_bytes=128,
        policy="lease",
        lease=3,
        drop_threshold=0,
        prefill_cache_reuse=True,
        resident_shallow_layers=2,
    )
    specs = (ExpertPayloadSpec("weight", (2, 1), torch.float32),)
    shallow = SimpleNamespace(layer_id=1)
    offloaded = SimpleNamespace(layer_id=2)

    assert manager.reserve_layer(shallow, specs) is False
    assert not hasattr(shallow, "_moe_expert_offload_reserved")
    assert manager.reserve_layer(offloaded, specs) is True
    assert offloaded._moe_expert_offload_reserved is True


def test_global_stats_sum_traffic_but_take_shared_staging_peak():
    total = ExpertCacheStats(
        h2d_bytes=10, prefetch_h2d_bytes=1, staging_peak_bytes=100
    )
    total.add(
        ExpertCacheStats(
            h2d_bytes=20, prefetch_h2d_bytes=2, staging_peak_bytes=80
        )
    )
    total.add(
        ExpertCacheStats(
            h2d_bytes=30, prefetch_h2d_bytes=3, staging_peak_bytes=120
        )
    )
    assert total.h2d_bytes == 60
    assert total.prefetch_h2d_bytes == 6
    assert total.staging_peak_bytes == 120


def test_manager_records_batched_copy_submission(monkeypatch):
    manager = MoeExpertCacheManager(
        host_budget_bytes=1,
        gpu_budget_bytes=1,
        policy="lease",
        lease=3,
        drop_threshold=0,
        prefill_cache_reuse=True,
    )
    manager.copy_stream = object()
    submitted = []
    monkeypatch.setattr(
        expert_cache_module._CUDA_BATCH_COPY,
        "submit",
        lambda copies, stream, prefer_overlap=False: submitted.append(
            (copies, stream, prefer_overlap)
        )
        or True,
    )
    copies = [(torch.empty(1), torch.empty(1)) for _ in range(3)]
    stats = ExpertCacheStats()

    manager._submit_copies(copies, stats)

    assert submitted == [(copies, manager.copy_stream, False)]
    assert stats.batched_copy_submissions == 1
    assert stats.batched_copy_operations == 3
    assert stats.fallback_copy_operations == 0


def test_manager_falls_back_to_individual_copies(monkeypatch):
    manager = MoeExpertCacheManager(
        host_budget_bytes=1,
        gpu_budget_bytes=1,
        policy="lease",
        lease=3,
        drop_threshold=0,
        prefill_cache_reuse=True,
    )
    manager.copy_stream = object()
    monkeypatch.setattr(
        expert_cache_module._CUDA_BATCH_COPY,
        "submit",
        lambda copies, stream, prefer_overlap=False: False,
    )
    sources = [torch.tensor([1.0]), torch.tensor([2.0])]
    destinations = [torch.zeros(1), torch.zeros(1)]
    stats = ExpertCacheStats()

    manager._submit_copies(list(zip(destinations, sources)), stats)

    assert [tensor.item() for tensor in destinations] == [1.0, 2.0]
    assert stats.batched_copy_submissions == 0
    assert stats.batched_copy_operations == 0
    assert stats.fallback_copy_operations == 2


def test_server_args_rejects_conflicting_general_cpu_offload(monkeypatch):
    import sglang.srt.server_args as server_args_module

    args = server_args_module.ServerArgs(
        model_path="dummy",
        moe_expert_offload_gb=16,
        moe_runner_backend="flashinfer_cutlass",
        cpu_offload_gb=1,
    )
    monkeypatch.setattr(server_args_module, "is_cuda", lambda: True)
    monkeypatch.setattr(
        server_args_module.ServerArgs,
        "get_model_config",
        lambda self: SimpleNamespace(quantization="modelopt_fp4"),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        args._handle_moe_expert_offload()


@pytest.mark.parametrize("minimum_ratio", (0.0, -0.5, 1.01))
def test_server_args_rejects_invalid_prefetch_minimum_ratio(minimum_ratio):
    import sglang.srt.server_args as server_args_module

    args = server_args_module.ServerArgs(
        model_path="dummy",
        moe_expert_cache_prefetch_min_ratio=minimum_ratio,
    )
    with pytest.raises(
        ValueError, match="--moe-expert-cache-prefetch-min-ratio"
    ):
        args._handle_moe_expert_offload()


def test_server_args_normalizes_supported_expert_offload(monkeypatch):
    import sglang.srt.server_args as server_args_module

    args = server_args_module.ServerArgs(
        model_path="dummy",
        moe_expert_offload_gb=16,
        moe_expert_cache_gb=1,
        moe_runner_backend="flashinfer_cutlass",
    )
    monkeypatch.setattr(server_args_module, "is_cuda", lambda: True)
    monkeypatch.setattr(
        server_args_module.ServerArgs,
        "get_model_config",
        lambda self: SimpleNamespace(quantization="modelopt_fp4"),
    )
    args._handle_moe_expert_offload()
    assert args.disable_cuda_graph is True
    assert args.enable_torch_compile is False
    assert args.disable_shared_experts_fusion is True
    assert args.mem_fraction_static == 0.625
    assert args.max_total_tokens == 8192
    assert args.weight_loader_drop_cache_after_load is True
    assert args.moe_expert_cache_prefetch_min_ratio == 0.5
    assert json.loads(args.model_loader_extra_config)[
        "enable_multithread_load"
    ] is False


def test_server_args_accepts_symmetric_gptq_int4_expert_offload(monkeypatch):
    import sglang.srt.server_args as server_args_module

    args = server_args_module.ServerArgs(
        model_path="dummy",
        quantization="moe_wna16",
        moe_expert_offload_gb=16,
        moe_expert_cache_gb=1,
        moe_runner_backend="auto",
        cuda_graph_backend_decode="breakable",
    )
    monkeypatch.setattr(server_args_module, "is_cuda", lambda: True)
    monkeypatch.setattr(
        server_args_module.ServerArgs,
        "get_model_config",
        lambda self: SimpleNamespace(
            quantization="moe_wna16",
            hf_config=SimpleNamespace(
                quantization_config={
                    "quant_method": "gptq",
                    "bits": 4,
                    "sym": True,
                    "desc_act": False,
                }
            ),
        ),
    )
    args._handle_moe_expert_offload()
    assert args.disable_cuda_graph is True
    assert args.cuda_graph_backend_decode == "breakable"
    assert args.enable_torch_compile is False
    assert args.disable_shared_experts_fusion is True


def test_server_args_accepts_bs1_qwen35_bundled_mtp_expert_offload(monkeypatch):
    import sglang.srt.server_args as server_args_module

    model_path = "dummy"
    args = server_args_module.ServerArgs(
        model_path=model_path,
        quantization="moe_wna16",
        moe_expert_offload_gb=12.5,
        moe_expert_cache_gb=1.5,
        moe_runner_backend="auto",
        cuda_graph_backend_decode="breakable",
        max_running_requests=1,
        speculative_algorithm="NEXTN",
        speculative_draft_model_path=model_path,
        speculative_num_steps=2,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=3,
    )
    monkeypatch.setattr(server_args_module, "is_cuda", lambda: True)
    monkeypatch.setattr(
        server_args_module.ServerArgs,
        "get_model_config",
        lambda self: SimpleNamespace(
            quantization="moe_wna16",
            hf_config=SimpleNamespace(
                architectures=["Qwen3_5MoeForConditionalGeneration"],
                quantization_config={
                    "quant_method": "gptq",
                    "bits": 4,
                    "sym": True,
                    "desc_act": False,
                },
            ),
        ),
    )

    args._handle_moe_expert_offload()

    assert args.disable_cuda_graph is True
    assert args.cuda_graph_backend_decode == "breakable"
    assert args.enable_torch_compile is False


def test_server_args_rejects_batched_mtp_expert_offload(monkeypatch):
    import sglang.srt.server_args as server_args_module

    model_path = "dummy"
    args = server_args_module.ServerArgs(
        model_path=model_path,
        quantization="moe_wna16",
        moe_expert_offload_gb=12.5,
        moe_runner_backend="auto",
        max_running_requests=2,
        speculative_algorithm="NEXTN",
        speculative_draft_model_path=model_path,
        speculative_eagle_topk=1,
    )
    monkeypatch.setattr(server_args_module, "is_cuda", lambda: True)
    monkeypatch.setattr(
        server_args_module.ServerArgs,
        "get_model_config",
        lambda self: SimpleNamespace(
            quantization="moe_wna16",
            hf_config=SimpleNamespace(
                architectures=["Qwen3_5MoeForConditionalGeneration"],
                quantization_config={
                    "quant_method": "gptq",
                    "bits": 4,
                    "sym": True,
                    "desc_act": False,
                },
            ),
        ),
    )

    with pytest.raises(ValueError, match="single-request Qwen3.5"):
        args._handle_moe_expert_offload()


def test_server_args_rejects_asymmetric_gptq_expert_offload(monkeypatch):
    import sglang.srt.server_args as server_args_module

    args = server_args_module.ServerArgs(
        model_path="dummy",
        quantization="moe_wna16",
        moe_expert_offload_gb=16,
        moe_runner_backend="auto",
    )
    monkeypatch.setattr(server_args_module, "is_cuda", lambda: True)
    monkeypatch.setattr(
        server_args_module.ServerArgs,
        "get_model_config",
        lambda self: SimpleNamespace(
            quantization="moe_wna16",
            hf_config=SimpleNamespace(
                quantization_config={
                    "quant_method": "gptq",
                    "bits": 4,
                    "sym": False,
                    "desc_act": False,
                }
            ),
        ),
    )
    with pytest.raises(ValueError, match="symmetric 4-bit GPTQ"):
        args._handle_moe_expert_offload()
