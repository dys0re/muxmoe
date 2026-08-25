from __future__ import annotations

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_prefill_initial_state_is_contiguous_fp32():
    state = torch.arange(48, dtype=torch.bfloat16).reshape(3, 2, 2, 4)
    indices = torch.tensor([2, 0], dtype=torch.int64)

    prepared = FlashInferGDNKernel._prepare_prefill_initial_state(state, indices)

    assert prepared.dtype == torch.float32
    assert prepared.is_contiguous()
    torch.testing.assert_close(prepared, state[indices].float())
