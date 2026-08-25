"""Regression test for GemmaRMSNorm weight loading under CPU parameter offload.

``GemmaRMSNorm`` precomputes ``weight + 1`` into ``gemma_weight``, a
*non-persistent* buffer that the fused rmsnorm / allreduce-fusion kernels read
directly. ``OffloaderV1`` (``--cpu-offload-gb``) walks ``module.parameters()``
only, so it moves ``weight`` to host memory and leaves ``gemma_weight`` on the
compute device. The loader then ran ``torch.add(param.data, 1.0,
out=self.gemma_weight)`` across the device boundary and died with
``RuntimeError: Expected all tensors to be on the same device`` partway through
loading a checkpoint.

Moving the buffer alongside the parameter is not an option: it is
non-persistent, so it is absent from ``state_dict()`` and the offloader's
forward hook — which restores tensors via ``functional_call(module,
state_dict)`` — could never bring it back to the device. The buffer must stay
resident (it is only ``hidden_size`` elements) and the add must happen on its
device.

Needs two devices to be meaningful: with everything on CPU the ``.to()`` in the
fix is a no-op and a broken implementation would pass.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.test.test_utils import CustomTestCase

HIDDEN_SIZE = 16


@unittest.skipUnless(torch.cuda.is_available(), "needs a second device")
class TestGemmaRMSNormOffloadedWeightLoad(CustomTestCase):
    """The precomputed buffer must track the parameter across a device split."""

    def _make_norm_with_offloaded_weight(self):
        """A norm whose parameter sits on CPU while its buffer stays on CUDA."""
        norm = GemmaRMSNorm(HIDDEN_SIZE).to("cuda")
        # Exactly what OffloaderV1.maybe_offload_to_cpu does: parameters only.
        for param in norm.parameters():
            param.data = param.data.to("cpu")
        self.assertEqual(norm.weight.device.type, "cpu")
        self.assertEqual(norm.gemma_weight.device.type, "cuda")
        return norm

    def test_load_across_device_split(self):
        norm = self._make_norm_with_offloaded_weight()
        loaded = torch.arange(HIDDEN_SIZE, dtype=norm.weight.dtype)

        # Raised RuntimeError("Expected all tensors to be on the same device")
        # before the fix.
        norm.weight.weight_loader(norm.weight, loaded)

        self.assertEqual(norm.weight.device.type, "cpu")
        torch.testing.assert_close(norm.weight.data, loaded)

    def test_buffer_stays_resident_and_holds_weight_plus_one(self):
        norm = self._make_norm_with_offloaded_weight()
        buffer_ptr = norm.gemma_weight.data_ptr()
        loaded = torch.arange(HIDDEN_SIZE, dtype=norm.weight.dtype)

        norm.weight.weight_loader(norm.weight, loaded)

        # Fused kernels capture this buffer, so its storage must not be rebound.
        self.assertEqual(norm.gemma_weight.data_ptr(), buffer_ptr)
        self.assertEqual(norm.gemma_weight.device.type, "cuda")
        torch.testing.assert_close(norm.gemma_weight.cpu(), loaded + 1.0)

    def test_single_device_load_is_unchanged(self):
        """The common path keeps both tensors on the device; ``.to()`` is a no-op."""
        norm = GemmaRMSNorm(HIDDEN_SIZE).to("cuda")
        loaded = torch.arange(HIDDEN_SIZE, dtype=norm.weight.dtype, device="cuda")

        norm.weight.weight_loader(norm.weight, loaded)

        self.assertEqual(norm.weight.device.type, "cuda")
        torch.testing.assert_close(norm.weight.data, loaded)
        torch.testing.assert_close(norm.gemma_weight, loaded + 1.0)


if __name__ == "__main__":
    unittest.main()
