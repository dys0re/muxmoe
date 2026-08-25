import pytest
import torch

from sglang.srt.layers.moe.expert_residency import (
    ExpertResidencyManager,
    ExpertWeightBundle,
)


def _bundle(expert_id: int) -> ExpertWeightBundle:
    return ExpertWeightBundle(
        {
            "w13_weight": torch.full((4, 3), expert_id, dtype=torch.uint8),
            "w13_weight_scale": torch.full((2, 3), expert_id, dtype=torch.float32),
            "w2_weight": torch.full((3, 2), expert_id, dtype=torch.uint8),
        },
        metadata={"quant_format": "test-packed"},
    )


def test_residency_keeps_quantized_tensor_contract_and_maps_ids():
    manager = ExpertResidencyManager(max_resident_experts=2)
    manager.register_layer(7, [_bundle(0), _bundle(1), _bundle(2)])

    resident = manager.ensure_resident(7, torch.tensor([2, 2, 0]))

    assert resident.expert_ids.tolist() == [2, 0]
    assert resident.id_to_slot == {2: 0, 0: 1}
    assert resident.weights["w13_weight"].dtype == torch.uint8
    assert resident.weights["w13_weight_scale"].shape == (2, 2, 3)
    assert resident.weights["w13_weight"][:, 0, 0].tolist() == [2, 0]


def test_residency_reuses_fixed_slots_and_evicts_lru():
    manager = ExpertResidencyManager(max_resident_experts=2)
    manager.register_layer(0, [_bundle(0), _bundle(1), _bundle(2)])

    first = manager.ensure_resident(0, [0, 1])
    second = manager.ensure_resident(0, [1, 2])

    assert first.weights["w13_weight"][:, 0, 0].tolist() == [0, 1]
    assert second.weights["w13_weight"][:, 0, 0].tolist() == [1, 2]
    assert manager.cached_expert_count == 2


    manager = ExpertResidencyManager(max_resident_experts=2)
    altered = ExpertWeightBundle(
        {
            "w13_weight": torch.zeros((5, 3), dtype=torch.uint8),
            "w2_weight": torch.zeros((3, 2), dtype=torch.uint8),
        }
    )

    with pytest.raises(ValueError, match="different tensor contract"):
        manager.register_layer(0, [_bundle(0), altered])


def test_residency_rejects_over_capacity_and_unknown_experts():
    manager = ExpertResidencyManager(max_resident_experts=1)
    manager.register_layer(0, [_bundle(0), _bundle(1)])

    with pytest.raises(RuntimeError, match="exceeding"):
        manager.ensure_resident(0, [0, 1])
    with pytest.raises(IndexError, match="out-of-range"):
        manager.ensure_resident(0, [2])


def test_residency_requires_cpu_bundles():
    with pytest.raises(ValueError, match="stored on CPU"):
        ExpertWeightBundle({"weight": torch.empty(2, device="meta")})
