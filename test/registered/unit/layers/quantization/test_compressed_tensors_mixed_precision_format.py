"""CPU regression test for compressed-tensors checkpoints in "mixed-precision" format.

compressed-tensors reports ``format: "mixed-precision"`` at the top level when the
config groups disagree, and then carries the real format on each group (e.g.
``float-quantized`` for FP8 attention projections, ``nvfp4-pack-quantized`` for
NVFP4 experts). ``_quantization_scheme_map_from_config`` used to test only the
top-level format against ``is_activation_quantization_format``. "mixed-precision"
is not in that list, so the ``input_activations`` of *every* group were silently
left as ``None``.

The damage showed up far from the cause: with ``input_quant=None`` the activation
predicates (``_is_fp4a4_nvfp4``, ``_is_fp8_w8a8``) all return False, so
``get_moe_scheme`` fell through to ``_is_dynamic_token_w8a8`` and died on
``AttributeError: 'NoneType' object has no attribute 'num_bits'`` while building
the model. ``_get_scheme_from_parts`` gated the whole activation-quantized linear
branch on the same top-level format.

These tests pin the contract: each target records its own group's format, and the
activation args survive parsing for every group. This is pure config parsing (no
weights are created and no kernels run), so it runs on CPU.

The config mirrors the real Qwen3.6-35B-A3B-NVFP4 quant config.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.test.test_utils import CustomTestCase

ATTN_TARGET = r"re:.*self_attn\.(q|k|v|o)_proj$"
EXPERT_TARGET = r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$"

# FP8 attention projections: channel-wise weights, dynamic per-token activations.
_FP8_GROUP = {
    "targets": [ATTN_TARGET],
    "format": "float-quantized",
    "weights": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "channel",
        "dynamic": False,
    },
    "input_activations": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "token",
        "dynamic": True,
    },
}

# NVFP4 experts: 4-bit tensor_group weights and activations, group size 16.
_NVFP4_GROUP = {
    "targets": [EXPERT_TARGET],
    "format": "nvfp4-pack-quantized",
    "weights": {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "tensor_group",
        "group_size": 16,
        "dynamic": False,
    },
    "input_activations": {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "tensor_group",
        "group_size": 16,
        "dynamic": "local",
    },
}


def _make_config(top_level_format, groups):
    return {
        "quant_method": "compressed-tensors",
        "format": top_level_format,
        "config_groups": groups,
        "ignore": ["lm_head", "re:.*mlp.gate$"],
    }


class TestMixedPrecisionActivationArgs(CustomTestCase):
    """Regression: per-group format must drive activation parsing."""

    def _parse_mixed(self):
        return CompressedTensorsConfig.from_config(
            _make_config(
                "mixed-precision", {"group_0": _FP8_GROUP, "group_1": _NVFP4_GROUP}
            )
        )

    def test_each_target_records_its_own_group_format(self):
        quant_config = self._parse_mixed()

        self.assertEqual(
            quant_config.target_scheme_map[ATTN_TARGET]["format"], "float-quantized"
        )
        self.assertEqual(
            quant_config.target_scheme_map[EXPERT_TARGET]["format"],
            "nvfp4-pack-quantized",
        )
        # The top-level marker is preserved as-is for callers that still read it.
        self.assertEqual(quant_config.quant_format, "mixed-precision")

    def test_activation_args_survive_parsing(self):
        """The actual bug: both groups used to come back with input_activations=None."""
        quant_config = self._parse_mixed()

        fp8_input = quant_config.target_scheme_map[ATTN_TARGET]["input_activations"]
        self.assertIsNotNone(fp8_input)
        self.assertEqual(fp8_input.num_bits, 8)
        self.assertEqual(fp8_input.strategy, "token")

        nvfp4_input = quant_config.target_scheme_map[EXPERT_TARGET]["input_activations"]
        self.assertIsNotNone(nvfp4_input)
        self.assertEqual(nvfp4_input.num_bits, 4)
        self.assertEqual(nvfp4_input.strategy, "tensor_group")
        self.assertEqual(nvfp4_input.group_size, 16)

    def test_scheme_predicates_recognize_both_groups(self):
        """With activations restored, the predicates that select a kernel fire again."""
        quant_config = self._parse_mixed()

        fp8 = quant_config.target_scheme_map[ATTN_TARGET]
        self.assertTrue(
            quant_config._is_fp8_w8a8(fp8["weights"], fp8["input_activations"])
        )

        nvfp4 = quant_config.target_scheme_map[EXPERT_TARGET]
        self.assertTrue(
            quant_config._is_fp4a4_nvfp4(nvfp4["weights"], nvfp4["input_activations"])
        )

    def test_uniform_format_still_parses_without_per_group_format(self):
        """A single-group checkpoint has no per-group format; the top level still wins."""
        group = {k: v for k, v in _FP8_GROUP.items() if k != "format"}
        quant_config = CompressedTensorsConfig.from_config(
            _make_config("float-quantized", {"group_0": group})
        )

        scheme = quant_config.target_scheme_map[ATTN_TARGET]
        self.assertEqual(scheme["format"], "float-quantized")
        self.assertIsNotNone(scheme["input_activations"])

    def test_weight_only_group_under_mixed_precision_keeps_none_activations(self):
        """A group that genuinely has no activation quant must stay None, not crash."""
        weight_only = {
            "targets": [EXPERT_TARGET],
            "format": "pack-quantized",
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 128,
            },
            "input_activations": None,
        }
        quant_config = CompressedTensorsConfig.from_config(
            _make_config(
                "mixed-precision", {"group_0": _FP8_GROUP, "group_1": weight_only}
            )
        )

        self.assertIsNone(
            quant_config.target_scheme_map[EXPERT_TARGET]["input_activations"]
        )
        self.assertIsNotNone(
            quant_config.target_scheme_map[ATTN_TARGET]["input_activations"]
        )


if __name__ == "__main__":
    unittest.main()
