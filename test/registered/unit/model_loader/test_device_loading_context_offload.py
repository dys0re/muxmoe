"""Regression test for device_loading_context under CPU parameter offload.

``load_weights_and_postprocess`` wraps every ``process_weights_after_loading``
call in ``device_loading_context`` so a CPU-offloaded module can be repacked on
the device and then pushed back to the host. The restore loop matched parameters
**by name**, and its trailing comment ("New parameters ... are untouched")
described the hole: a quantization method that renames while it repacks — e.g.
``CompressedTensorsW4A4Nvfp4MoE`` does

    layer.w13_weight = torch.nn.Parameter(layer.w13_weight_packed.data)
    delattr(layer, "w13_weight_packed")

— produces names absent from the entry snapshot, so those tensors stayed on the
device. The offload silently unwound itself exactly for the largest weights, and
loading Qwen3.6-35B-A3B-NVFP4 with ``--cpu-offload-gb 12`` died of CUDA OOM with
~9.6 GiB of expert weights stranded on the GPU.

The fix keys the decision on storage identity rather than name: storages already
resident at entry are left alone, and anything else found on the device at exit
is copied back to pinned host memory.

Needs CUDA: with one device the host/device split under test does not exist.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.model_loader.loader import device_loading_context
from sglang.test.test_utils import CustomTestCase

CUDA = torch.device("cuda")


class _Repacking(torch.nn.Module):
    """Stands in for a quantized MoE layer that repacks on the device."""

    def __init__(self):
        super().__init__()
        self.weight_packed = torch.nn.Parameter(
            torch.arange(8, dtype=torch.float32), requires_grad=False
        )

    def offload_to_cpu(self):
        """What OffloaderV1.maybe_offload_to_cpu does: parameters only."""
        for param in self.parameters():
            param.data = param.data.to("cpu")
        return self

    def process_weights_after_loading(self):
        # The rename that used to defeat the restore loop.
        self.weight = torch.nn.Parameter(self.weight_packed.data * 2, requires_grad=False)
        delattr(self, "weight_packed")


class _RepackingWithBroadcastScale(_Repacking):
    """Publishes an expanded scalar, as the TRT-LLM NVFP4 MoE repack does."""

    def process_weights_after_loading(self):
        super().process_weights_after_loading()
        # `expand` gives stride 0; copying that layout verbatim would alias.
        self.scale = torch.nn.Parameter(
            self.weight.data.min().expand(8), requires_grad=False
        )


@unittest.skipUnless(torch.cuda.is_available(), "needs a host/device split")
class TestDeviceLoadingContextOffload(CustomTestCase):

    def test_renamed_parameter_returns_to_host(self):
        module = _Repacking().to(CUDA).offload_to_cpu()

        with device_loading_context(module, CUDA):
            self.assertEqual(module.weight_packed.device.type, "cuda")
            module.process_weights_after_loading()

        # Stayed on cuda before the fix, stranding the largest weights there.
        self.assertEqual(module.weight.device.type, "cpu")
        self.assertFalse(hasattr(module, "weight_packed"))
        torch.testing.assert_close(
            module.weight.data, torch.arange(8, dtype=torch.float32) * 2
        )

    def test_unrenamed_parameter_still_returns_to_host(self):
        """The path that already worked must keep working."""
        module = _Repacking().to(CUDA).offload_to_cpu()

        with device_loading_context(module, CUDA):
            module.weight_packed.data.add_(1.0)

        self.assertEqual(module.weight_packed.device.type, "cpu")
        torch.testing.assert_close(
            module.weight_packed.data, torch.arange(8, dtype=torch.float32) + 1.0
        )

    def test_resident_module_is_left_on_device(self):
        """A module that was never offloaded must not be dragged to the host."""
        module = _Repacking().to(CUDA)

        with device_loading_context(module, CUDA):
            module.process_weights_after_loading()

        self.assertEqual(module.weight.device.type, "cuda")

    def test_resident_parameter_of_partially_offloaded_module_stays(self):
        """OffloaderV1 offloads per parameter, so a module can be split."""
        module = _Repacking().to(CUDA)
        module.resident = torch.nn.Parameter(
            torch.ones(4, device=CUDA), requires_grad=False
        )
        module.weight_packed.data = module.weight_packed.data.to("cpu")
        resident_ptr = module.resident.data_ptr()

        with device_loading_context(module, CUDA):
            module.process_weights_after_loading()

        self.assertEqual(module.weight.device.type, "cpu")
        self.assertEqual(module.resident.device.type, "cuda")
        # Untouched means the same storage, not merely the same values.
        self.assertEqual(module.resident.data_ptr(), resident_ptr)

    def test_broadcast_scale_published_by_repack(self):
        """A repack may hand back an expanded scalar; copying it must not alias."""
        module = _RepackingWithBroadcastScale().to(CUDA).offload_to_cpu()

        with device_loading_context(module, CUDA):
            module.process_weights_after_loading()

        # Raised RuntimeError("more than one element of the written-to tensor
        # refers to a single memory location") before the fix.
        self.assertEqual(module.scale.device.type, "cpu")
        self.assertEqual(module.scale.shape, (8,))
        torch.testing.assert_close(module.scale.data, torch.zeros(8))


if __name__ == "__main__":
    unittest.main()
