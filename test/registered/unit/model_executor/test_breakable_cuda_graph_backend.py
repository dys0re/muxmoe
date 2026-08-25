import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)


def test_breakable_backend_buffers_dataclass_outputs():
    backend = object.__new__(BreakableCudaGraphBackend)
    output = LogitsProcessorOutput(
        next_token_logits=torch.arange(8).reshape(2, 4),
        hidden_states=torch.arange(6).reshape(2, 3),
    )

    buffer = backend._alloc_full_buffer(output, 4)
    assert isinstance(buffer, LogitsProcessorOutput)
    assert buffer.next_token_logits.shape == (4, 4)
    assert buffer.hidden_states.shape == (4, 3)

    backend._copy_output_to_buffer(output, buffer, 2)
    sliced = backend._slice_output(buffer, 2)
    torch.testing.assert_close(sliced.next_token_logits, output.next_token_logits)
    torch.testing.assert_close(sliced.hidden_states, output.hidden_states)


def test_breakable_backend_uses_speculative_output_rows_not_graph_key_size():
    backend = object.__new__(BreakableCudaGraphBackend)
    output = LogitsProcessorOutput(
        next_token_logits=torch.zeros(3, 16),
        hidden_states=torch.zeros(3, 8),
    )

    # A bs=1 target-verify graph with three proposed tokens is keyed by 1 but
    # must preserve all three output rows.
    assert backend._infer_output_rows(output) == 3
    assert backend._output_rows(output, cap=3) == 3

    buffer = backend._alloc_full_buffer(output, backend._infer_output_rows(output))
    assert buffer.next_token_logits.shape == (3, 16)
    assert buffer.hidden_states.shape == (3, 8)
