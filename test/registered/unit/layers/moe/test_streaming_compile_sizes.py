from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (
    _streaming_compile_sizes,
)


def test_streaming_compile_sizes_deduplicates_and_orders():
    assert _streaming_compile_sizes([512, 128, 512, 256], "inductor") == [
        128,
        256,
        512,
    ]


def test_streaming_compile_sizes_eager_warms_smallest_only():
    assert _streaming_compile_sizes([512, 128, 256], "eager") == [128]


def test_streaming_compile_sizes_handles_empty_input():
    assert _streaming_compile_sizes([], "eager") == []
