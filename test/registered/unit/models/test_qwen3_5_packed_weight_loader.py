"""
Unit tests for Qwen3_5GatedDeltaNet._make_packed_weight_loader.

Validates that per-tensor FP8 scales (scalar or single-element tensors)
are broadcast to every logical shard, while normal multi-element weights
are split correctly.

Regression test for https://github.com/sgl-project/sglang/issues/23051
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.models.qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5MoeForCausalLM,
)


def _make_mock_module(output_sizes):
    """Create a lightweight mock module with the attributes needed by the loader."""
    return SimpleNamespace(output_sizes=output_sizes)


def _make_per_tensor_scale_param(num_shards):
    """Create a PerTensorScaleParameter pre-allocated for `num_shards` scales.

    PerTensorScaleParameter requires a weight_loader callable;
    we supply a no-op since the packed loader wraps it anyway.
    """
    return PerTensorScaleParameter(
        data=torch.zeros(num_shards),
        weight_loader=lambda *args, **kwargs: None,
    )


class TestMakePackedWeightLoader(unittest.TestCase):
    """Tests for _make_packed_weight_loader broadcast / split logic."""

    # ------------------------------------------------------------------ #
    #  Per-tensor scale broadcast                                         #
    # ------------------------------------------------------------------ #

    def test_scalar_weight_broadcast(self):
        """A 0-d scalar should be broadcast (via .view(-1)) to every shard."""
        module = _make_mock_module(output_sizes=[128, 128, 64, 64])
        param = _make_per_tensor_scale_param(num_shards=4)

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        scalar = torch.tensor(0.5)  # shape=[]
        loader(param, scalar, loaded_shard_id=(0, 1, 2))

        self.assertEqual(len(calls), 3)
        for shard_id, chunk in calls:
            self.assertEqual(chunk.shape, torch.Size([1]))
            self.assertAlmostEqual(chunk.item(), 0.5, places=5)

    def test_single_element_tensor_broadcast(self):
        """A [1]-shaped tensor (e.g. per-tensor weight_scale) should be
        broadcast to every logical shard."""
        module = _make_mock_module(output_sizes=[128, 128, 64, 64])
        param = _make_per_tensor_scale_param(num_shards=4)

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        scale = torch.tensor([0.25])  # shape=[1]
        loader(param, scale, loaded_shard_id=(0, 1, 2))

        self.assertEqual(len(calls), 3)
        for idx, (shard_id, chunk) in enumerate(calls):
            self.assertEqual(shard_id, idx)
            self.assertEqual(chunk.shape, torch.Size([1]))
            self.assertAlmostEqual(chunk.item(), 0.25, places=5)

    def test_broadcast_with_two_shards(self):
        """Broadcast for in_proj_ba style (2 shards: b, a)."""
        module = _make_mock_module(output_sizes=[16, 16])
        param = _make_per_tensor_scale_param(num_shards=2)

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        scale = torch.tensor([0.1])
        loader(param, scale, loaded_shard_id=(0, 1))

        self.assertEqual(len(calls), 2)
        for shard_id, chunk in calls:
            self.assertEqual(chunk.shape, torch.Size([1]))
            self.assertAlmostEqual(chunk.item(), 0.1, places=5)

    # ------------------------------------------------------------------ #
    #  Normal weight split                                                #
    # ------------------------------------------------------------------ #

    def test_normal_weight_split(self):
        """Multi-element weights should be split by output_sizes, not broadcast."""
        module = _make_mock_module(output_sizes=[128, 128, 64])
        param = MagicMock()
        param.output_dim = 0

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        # Simulate a checkpoint weight that covers shard 0, 1, 2
        weight = torch.randn(128 + 128 + 64, 256)
        loader(param, weight, loaded_shard_id=(0, 1, 2))

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][1].shape[0], 128)
        self.assertEqual(calls[1][1].shape[0], 128)
        self.assertEqual(calls[2][1].shape[0], 64)

    # ------------------------------------------------------------------ #
    #  Passthrough for non-tuple shard_id                                 #
    # ------------------------------------------------------------------ #

    def test_int_shard_id_passthrough(self):
        """An int shard_id should bypass the tuple logic entirely."""
        module = _make_mock_module(output_sizes=[128, 128, 64, 64])

        calls = []

        def original_loader(p, loaded_weight, shard_id):
            calls.append(("original", shard_id))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        weight = torch.randn(128, 256)
        loader(MagicMock(), weight, loaded_shard_id=2)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ("original", 2))

    def test_none_shard_id_passthrough(self):
        """None shard_id should pass through to the original loader."""
        module = _make_mock_module(output_sizes=[128])

        calls = []

        def original_loader(p, loaded_weight, shard_id):
            calls.append(("original", shard_id))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        weight = torch.randn(128, 256)
        loader(MagicMock(), weight, loaded_shard_id=None)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ("original", None))

    # ------------------------------------------------------------------ #
    #  Edge case: nested single-element tensors                           #
    # ------------------------------------------------------------------ #

    def test_nested_single_element_tensor_broadcast(self):
        """A [[value]] shaped tensor (numel==1, ndim==2) should also broadcast."""
        module = _make_mock_module(output_sizes=[128, 128, 64])
        param = _make_per_tensor_scale_param(num_shards=3)

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        scale = torch.tensor([[0.75]])  # shape=[1,1], numel==1
        loader(param, scale, loaded_shard_id=(0, 1, 2))

        self.assertEqual(len(calls), 3)
        for shard_id, chunk in calls:
            # .view(-1) should flatten to [1]
            self.assertEqual(chunk.shape, torch.Size([1]))
            self.assertAlmostEqual(chunk.item(), 0.75, places=5)


class TestQwen35MoeRouterConfig(unittest.TestCase):
    def test_missing_norm_topk_prob_defaults_to_true(self):
        config = SimpleNamespace(norm_topk_prob=None)

        with unittest.mock.patch.object(
            Qwen3_5MoeForCausalLM.__mro__[1], "__init__", return_value=None
        ):
            Qwen3_5MoeForCausalLM(config)

        self.assertIs(config.norm_topk_prob, True)

    def test_explicit_norm_topk_prob_is_preserved(self):
        config = SimpleNamespace(norm_topk_prob=False)

        with unittest.mock.patch.object(
            Qwen3_5MoeForCausalLM.__mro__[1], "__init__", return_value=None
        ):
            Qwen3_5MoeForCausalLM(config)

        self.assertIs(config.norm_topk_prob, False)


class TestQwen35GatedDeltaNetAliases(unittest.TestCase):
    def test_refreshes_conv_weight_view_after_storage_replacement(self):
        layer = Qwen3_5GatedDeltaNet.__new__(Qwen3_5GatedDeltaNet)
        torch.nn.Module.__init__(layer)

        layer.conv1d = torch.nn.Module()
        layer.conv1d.register_parameter(
            "weight", torch.nn.Parameter(torch.arange(24).view(6, 1, 4).float())
        )
        layer.conv1d.register_parameter("bias", None)
        layer.attn = SimpleNamespace(
            conv_weights=torch.full((6, 4), float("nan")),
        )

        layer._refresh_attn_conv_weight_view()

        self.assertTrue(
            torch.equal(layer.attn.conv_weights, layer.conv1d.weight.view(6, 4))
        )
        self.assertEqual(
            layer.attn.conv_weights.untyped_storage().data_ptr(),
            layer.conv1d.weight.untyped_storage().data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
