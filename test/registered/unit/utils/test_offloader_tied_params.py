"""Regression test for OffloaderV1's forward hook with tied parameters.

``OffloaderV1`` restores an offloaded module for a forward by copying every
``state_dict()`` entry to the device and handing the result to
``functional_call``. A tied parameter appears under several names, and the
per-name ``.to(device)`` produced an independent copy for each, so
``functional_call`` saw tied keys that disagreed on identity and refused:

    ValueError: functional_call got multiple values for keys
    ['linear_attn.attn.dt_bias', 'linear_attn.dt_bias'], which are tied.

Qwen3.6-35B-A3B exposes the GatedDeltaNet ``dt_bias`` under both names, so every
offloaded linear-attention layer hit this on the first forward — after a
four-minute weight load.

The fix maps each source storage to one device tensor, which keeps the tie intact
and drops the duplicate host-to-device copy.

Needs CUDA: with everything already on one device the copy is a no-op and a
broken implementation would pass.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.utils.offloader import OffloaderV1
from sglang.test.test_utils import CustomTestCase


class _Inner(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias


class _TiedBias(torch.nn.Module):
    """Mirrors the GatedDeltaNet layout: one bias reachable by two names."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        # Same Parameter object under a second name -> tied in state_dict().
        self.attn = _Inner(self.bias)

    def forward(self, x):
        return x + self.bias + self.attn.bias


@unittest.skipUnless(torch.cuda.is_available(), "needs a host/device split")
class TestOffloaderV1TiedParameters(CustomTestCase):

    def _offloaded_module(self):
        module = _TiedBias().to("cuda")
        names = {name for name, _ in module.state_dict().items()}
        self.assertEqual(names, {"bias", "attn.bias"}, "test needs a tied pair")

        offloader = OffloaderV1(cpu_offload_max_bytes=1 << 30)
        module = offloader.maybe_offload_to_cpu(module)
        self.assertEqual(module.bias.device.type, "cpu")
        return module

    def test_forward_with_tied_parameter(self):
        module = self._offloaded_module()
        x = torch.zeros(4, device="cuda")

        # Raised ValueError("... which are tied") before the fix.
        out = module(x)

        self.assertEqual(out.device.type, "cuda")
        torch.testing.assert_close(out, torch.full((4,), 2.0, device="cuda"))

    def test_forward_is_repeatable(self):
        """The hook reinstalls itself, so a second call must behave identically."""
        module = self._offloaded_module()
        x = torch.zeros(4, device="cuda")

        first = module(x)
        second = module(x)

        torch.testing.assert_close(first, second)
        self.assertEqual(module.bias.device.type, "cpu")

    def test_untied_parameters_are_not_merged(self):
        """Distinct storages must still get their own device tensors."""

        class _TwoBiases(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.ones(4), requires_grad=False)
                self.b = torch.nn.Parameter(
                    torch.full((4,), 3.0), requires_grad=False
                )

            def forward(self, x):
                return x + self.a * 10 + self.b

        module = OffloaderV1(cpu_offload_max_bytes=1 << 30).maybe_offload_to_cpu(
            _TwoBiases().to("cuda")
        )

        out = module(torch.zeros(4, device="cuda"))

        torch.testing.assert_close(out, torch.full((4,), 13.0, device="cuda"))


if __name__ == "__main__":
    unittest.main()
